from __future__ import annotations

from datetime import timedelta
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from betting_data import get_db_status, load_upcoming_ws_odds


ORBITX_MARKET_URL_TEMPLATE = (
    "https://www.orbitxch.com/customer/sport/1/market/{market_id}"
)
MIN_EV_DEFAULT = 2.0

OUTCOMES = (
    {
        "key": "home",
        "label": "Home",
        "name_col": "home_name",
        "team_col": "HomeTeam_clean",
        "fallback_team_col": "HomeTeam",
        "prob_col": "p_calib_home",
        "pred_col": "home_pred",
        "max_col": "home_max",
    },
    {
        "key": "draw",
        "label": "Draw",
        "name_col": "draw_name",
        "team_col": None,
        "fallback_team_col": None,
        "prob_col": "p_calib_draw",
        "pred_col": "draw_pred",
        "max_col": "draw_max",
    },
    {
        "key": "away",
        "label": "Away",
        "name_col": "away_name",
        "team_col": "AwayTeam_clean",
        "fallback_team_col": "AwayTeam",
        "prob_col": "p_calib_away",
        "pred_col": "away_pred",
        "max_col": "away_max",
    },
)


def _as_float(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values > 0)


def _first_text(row: pd.Series, *columns: str | None, fallback: str = "-") -> str:
    for column in columns:
        if not column or column not in row.index:
            continue
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return fallback


def _fmt_odd(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{float(numeric):.2f}" if pd.notna(numeric) else "-"


def _fmt_size(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or float(numeric) <= 0:
        return ""
    number = float(numeric)
    if number >= 1000:
        return f"{number / 1000:.1f}k"
    return f"{number:.0f}"


def _fmt_pct(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{float(numeric):+.1f}%" if pd.notna(numeric) else "-"


def _safe_int(value: object) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    return int(float(numeric)) if pd.notna(numeric) else 0


def _fmt_identifier(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except Exception:
        return str(value).strip()
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.9f}".rstrip("0").rstrip(".")


def _fmt_ts(value: object, seconds: bool = False) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "-"
    return timestamp.strftime("%d/%m %H:%M:%S" if seconds else "%d/%m %H:%M")


def _orbitx_url(market_id: object) -> str | None:
    value = str(market_id or "").strip()
    if not value:
        return None
    return ORBITX_MARKET_URL_TEMPLATE.format(market_id=value)


def _fair_odds(probability: object, prediction: object) -> float:
    probability_value = pd.to_numeric(probability, errors="coerce")
    if pd.notna(probability_value) and 0 < float(probability_value) < 1:
        return 1.0 / float(probability_value)

    prediction_value = pd.to_numeric(prediction, errors="coerce")
    if pd.notna(prediction_value) and 1.01 <= float(prediction_value) <= 100:
        return float(prediction_value)

    return np.nan


def _ev_back(back_odds: object, fair_odds: object) -> float:
    back = pd.to_numeric(back_odds, errors="coerce")
    fair = pd.to_numeric(fair_odds, errors="coerce")
    if pd.isna(back) or pd.isna(fair) or float(back) <= 0 or float(fair) <= 0:
        return np.nan
    return (float(back) / float(fair) - 1.0) * 100.0


def _ev_lay(lay_odds: object, fair_odds: object) -> float:
    lay = pd.to_numeric(lay_odds, errors="coerce")
    fair = pd.to_numeric(fair_odds, errors="coerce")
    if pd.isna(lay) or pd.isna(fair) or float(lay) <= 0 or float(fair) <= 0:
        return np.nan
    return (float(fair) / float(lay) - 1.0) * 100.0


def _ev_class(value: object, min_ev: float) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    if float(numeric) >= max(10.0, min_ev):
        return " ws-ev-strong"
    if float(numeric) >= min_ev:
        return " ws-ev-good"
    if float(numeric) >= 0:
        return " ws-ev-neutral"
    return " ws-ev-bad"


def _sort_options(values: list[str]) -> list[str]:
    return sorted(
        [value for value in values if str(value).strip()], key=lambda item: item.lower()
    )


def _date_group_label(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "Horaire inconnu"
    return timestamp.strftime("%d/%m/%Y %H:%M")


def _prepare_matches(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["updated_at"] = pd.to_datetime(prepared.get("updated_at"), errors="coerce")
    prepared["match_date"] = pd.to_datetime(
        prepared.get("feed_match_date"), errors="coerce"
    )
    prepared["status"] = prepared.get("status", "").astype(str).str.upper()
    prepared["status"] = prepared["status"].replace(
        {"": "PRED", "NAN": "PRED", "NONE": "PRED"}
    )
    prepared["has_ws_odds"] = (
        pd.to_numeric(prepared.get("has_ws_odds"), errors="coerce")
        .fillna(0)
        .astype(int)
        .eq(1)
    )
    prepared["inplay_flag"] = (
        pd.to_numeric(prepared.get("inplay"), errors="coerce").fillna(0).astype(int)
    )
    prepared["is_inplay_effective"] = prepared["inplay_flag"].eq(1)
    prepared["League"] = (
        prepared.get("LeagueName", "Ligue inconnue")
        .fillna("Ligue inconnue")
        .astype(str)
    )

    for outcome in OUTCOMES:
        key = outcome["key"]
        back_cols = [f"{key}_back", f"{key}_back_1", f"{key}_back_2"]
        lay_cols = [f"{key}_lay", f"{key}_lay_1", f"{key}_lay_2"]
        prepared[f"best_{key}_back"] = (
            prepared[[c for c in back_cols if c in prepared.columns]]
            .apply(_as_float)
            .max(axis=1)
        )
        prepared[f"best_{key}_lay"] = (
            prepared[[c for c in lay_cols if c in prepared.columns]]
            .apply(_as_float)
            .min(axis=1)
        )
        prepared[f"fair_{key}"] = prepared.apply(
            lambda row, current=outcome: _fair_odds(
                row.get(current["prob_col"]), row.get(current["pred_col"])
            ),
            axis=1,
        )
        prepared[f"ev_{key}_back"] = prepared.apply(
            lambda row, current_key=key: _ev_back(
                row.get(f"best_{current_key}_back"), row.get(f"fair_{current_key}")
            ),
            axis=1,
        )
        prepared[f"ev_{key}_lay"] = prepared.apply(
            lambda row, current_key=key: _ev_lay(
                row.get(f"best_{current_key}_lay"), row.get(f"fair_{current_key}")
            ),
            axis=1,
        )

    ev_columns = [
        f"ev_{outcome['key']}_{side}"
        for outcome in OUTCOMES
        for side in ("back", "lay")
    ]
    prepared["best_ev"] = prepared[ev_columns].max(axis=1, skipna=True)
    prepared["best_ev"] = prepared["best_ev"].replace([-np.inf, np.inf], np.nan)
    prepared["opportunities_count"] = (
        prepared[ev_columns].ge(MIN_EV_DEFAULT).sum(axis=1)
    )
    prepared["match_label"] = prepared.apply(
        lambda row: (
            f"{_first_text(row, 'HomeTeam_clean', 'HomeTeam', 'link_home_name', 'home_name', fallback='Home')} - {_first_text(row, 'AwayTeam_clean', 'AwayTeam', 'link_away_name', 'away_name', fallback='Away')}"
        ),
        axis=1,
    )
    prepared["market_url"] = prepared["ID_MARKET"].apply(_orbitx_url)
    return prepared.sort_values(
        ["match_date", "updated_at"], ascending=[True, False], na_position="last"
    )


def _price_cell(
    row: pd.Series,
    key: str,
    column_suffix: str,
    side: str,
    best_value: object | None = None,
) -> str:
    odd_col = f"{key}_{column_suffix}"
    size_col = f"{odd_col}_size"
    odd = row.get(odd_col)
    cls = f"ws-{side}"
    if (
        best_value is not None
        and pd.notna(odd)
        and pd.notna(best_value)
        and float(odd) == float(best_value)
    ):
        cls += f" ws-best-{side}"
    return (
        f"<strong class='{cls}'>"
        f"<span class='ws-odd'>{_fmt_odd(odd)}</span>"
        f"<span class='ws-size'>{_fmt_size(row.get(size_col))}</span>"
        "</strong>"
    )


def _fair_cell(fair_odds: object, asian_max: object) -> str:
    max_label = _fmt_odd(asian_max)
    max_html = (
        f"<span class='ws-max-odd'>(AO max {max_label})</span>"
        if max_label != "-"
        else ""
    )
    return (
        "<span class='ws-chip'>"
        f"<span class='ws-odd'>{_fmt_odd(fair_odds)}</span>"
        f"{max_html}"
        "</span>"
    )


def _ids_line(row: pd.Series) -> str:
    id_parts = []
    for label, column in (
        ("Match", "feed_match_id"),
        ("Game", "GameId"),
        ("Market", "ID_MARKET"),
    ):
        value = _fmt_identifier(row.get(column))
        if value:
            id_parts.append(f"{label} {escape(value)}")
    if not id_parts:
        return ""
    return f"<div class='ws-ids'>{' | '.join(id_parts)}</div>"


def _outcome_name(row: pd.Series, outcome: dict[str, str | None]) -> str:
    if outcome["key"] == "draw":
        return _first_text(row, outcome["name_col"], fallback="Draw")
    return _first_text(
        row,
        outcome["team_col"],
        outcome["fallback_team_col"],
        f"link_{outcome['key']}_name" if outcome["key"] in {"home", "away"} else None,
        outcome["name_col"],
        fallback=str(outcome["label"]),
    )


def _outcome_row(row: pd.Series, outcome: dict[str, str | None], min_ev: float) -> str:
    key = str(outcome["key"])
    best_back = row.get(f"best_{key}_back")
    best_lay = row.get(f"best_{key}_lay")
    ev_back = row.get(f"ev_{key}_back")
    ev_lay = row.get(f"ev_{key}_lay")
    fair_odds = row.get(f"fair_{key}")
    asian_max = row.get(str(outcome.get("max_col") or f"{key}_max"))
    return f"""
        <div class='ws-grid-row'>
            <span class='p-col'>{escape(_outcome_name(row, outcome))}</span>
            {_price_cell(row, key, "back_1", "back", best_back)}
            {_price_cell(row, key, "back", "back", best_back)}
            {_price_cell(row, key, "lay", "lay", best_lay)}
            {_price_cell(row, key, "lay_1", "lay", best_lay)}
            {_fair_cell(fair_odds, asian_max)}
            <span class='ws-ev{_ev_class(ev_back, min_ev)}'>{_fmt_pct(ev_back)}</span>
            <span class='ws-ev{_ev_class(ev_lay, min_ev)}'>{_fmt_pct(ev_lay)}</span>
        </div>
    """


def _market_card_html(row: pd.Series, min_ev: float) -> str:
    has_ws_odds = bool(row.get("has_ws_odds", False))
    inplay_badge = "INPLAY" if bool(row.get("is_inplay_effective", False)) else "PRE"
    match_label = escape(str(row.get("match_label") or "-"))
    league = escape(str(row.get("League") or "Ligue inconnue"))
    status = escape(str(row.get("status") or "-"))
    kickoff = _fmt_ts(row.get("match_date"))
    updated = _fmt_ts(
        row.get("updated_at") if has_ws_odds else row.get("feed_maj"), seconds=True
    )
    market_url = row.get("market_url")
    best_ev = row.get("best_ev")
    has_opportunity = pd.notna(best_ev) and float(best_ev) >= min_ev
    opportunity_badge = (
        "BET" if has_opportunity else ("NO BET" if has_ws_odds else "PRED")
    )
    market_link = (
        f"<a href='{escape(str(market_url))}' target='_blank' rel='noopener noreferrer'>OrbitX</a>"
        if market_url
        else ""
    )
    analytics = ""
    if pd.notna(row.get("analytics_ev_pct")) or pd.notna(row.get("user_bets_count")):
        bets_count = _safe_int(row.get("user_bets_count"))
        analytics = (
            f"<div class='ws-meta'>Analytics EV {_fmt_pct(row.get('analytics_ev_pct'))} | "
            f"Paris utilisateur {bets_count}</div>"
        )

    rows = "".join(_outcome_row(row, outcome, min_ev) for outcome in OUTCOMES)
    ids_line = _ids_line(row)
    return f"""
<div class='ws-card'>
    <div class='ws-head'>
        <div class='ws-left'>
            <div class='ws-match'>{match_label}</div>
            {ids_line}
            <div class='ws-meta'>{league} | {status} | Debut {kickoff}</div>
            <div class='ws-meta'>{"Meilleure EV " + _fmt_pct(best_ev) + " | Updates " + str(_safe_int(row.get("n_updates"))) if has_ws_odds else "Predictions seules | Fair odds 1X2"}</div>
            {analytics}
        </div>
        <div class='ws-right'>
            <span class='ws-pill {"ws-pill-inplay" if inplay_badge == "INPLAY" else "ws-pill-pre"}'>{inplay_badge}</span>
            <span class='ws-pill {"ws-led-on" if has_opportunity else "ws-led-off"}'>{opportunity_badge}</span>
            <span class='ws-link'>{market_link}</span>
            <span class='ws-time'>Maj {updated}</span>
        </div>
    </div>
    <div class='ws-book'>
        <div class='ws-grid-head'>
            <span class='p-col'>Issue</span>
            <span>B1</span>
            <span>BB</span>
            <span>LB</span>
            <span>L1</span>
            <span>Fair</span>
            <span>EV B</span>
            <span>EV L</span>
        </div>
        {rows}
    </div>
</div>
"""


_WS_CSS = """
<style>
.ws-card {
    border: 1px solid rgba(16,35,63,0.10);
    background: rgba(255,255,255,0.92);
    border-radius: 14px;
    box-sizing: border-box;
    width: 100%;
    padding: clamp(8px, 1vw, 12px);
    margin-bottom: 12px;
    box-shadow: 0 12px 28px rgba(16,35,63,0.06);
}
.ws-head {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    align-items: flex-start;
    margin-bottom: 10px;
}
.ws-left { min-width: 0; }
.ws-match { font-weight: 800; color: #10233f; line-height: 1.25; }
.ws-ids { color: #64748b; font-size: 10px; font-weight: 700; margin-top: 2px; }
.ws-meta { color: #5e6d82; font-size: 11px; margin-top: 2px; }
.ws-right { display: flex; gap: 7px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
.ws-pill { border-radius: 999px; font-size: 10px; padding: 2px 8px; background: rgba(16,35,63,0.08); color: #10233f; font-weight: 700; }
.ws-pill-inplay { background: rgba(245,158,11,0.20); color: #7c3f00; border: 1px solid rgba(245,158,11,0.35); }
.ws-pill-pre { background: rgba(14,165,164,0.12); color: #0f766e; border: 1px solid rgba(14,165,164,0.24); }
.ws-led-on { background: rgba(34,197,94,0.16); color: #15803d; border: 1px solid rgba(34,197,94,0.35); }
.ws-led-off { background: rgba(148,163,184,0.16); color: #475569; border: 1px solid rgba(148,163,184,0.30); }
.ws-time { color: #5e6d82; font-size: 11px; }
.ws-link a { color: #0ea5a4; font-size: 11px; text-decoration: none; font-weight: 800; }
.ws-link a:hover { text-decoration: underline; }
.ws-book { background: rgba(16,35,63,0.03); border: 1px solid rgba(16,35,63,0.08); border-radius: 10px; padding: clamp(4px, 0.7vw, 7px); overflow-x: hidden; }
.ws-grid-head, .ws-grid-row { display: grid; grid-template-columns: minmax(76px, 1.35fr) repeat(7, minmax(0, 1fr)); gap: 2px; align-items: center; margin-bottom: 2px; width: 100%; min-width: 0; }
.ws-grid-head span { font-size: 10px; color: #5e6d82; font-weight: 800; text-align: center; }
.ws-grid-row .p-col { font-size: clamp(10px, 0.75vw, 12px); color: #10233f; text-align: left; font-weight: 800; padding-left: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
.ws-grid-row strong, .ws-chip, .ws-ev { text-align: center; font-size: clamp(10px, 0.78vw, 13px); border-radius: 6px; padding: clamp(3px, 0.45vw, 5px) 3px; min-height: 32px; display: flex; flex-direction: column; align-items: center; justify-content: center; line-height: 1.05; box-sizing: border-box; min-width: 0; overflow: hidden; }
.ws-size { font-size: clamp(8px, 0.65vw, 10px); font-weight: 500; opacity: 0.72; margin-top: 1px; }
.ws-max-odd { font-size: clamp(7px, 0.58vw, 9px); font-weight: 700; color: #64748b; margin-top: 1px; }
.ws-back { color: #10233f; border: 1px solid rgba(59,130,246,0.22); background: rgba(59,130,246,0.04); }
.ws-lay { color: #10233f; border: 1px solid rgba(244,114,182,0.24); background: rgba(244,114,182,0.04); }
.ws-best-back { background: #9bd3ff; border-color: rgba(37,99,235,0.36); font-weight: 800; }
.ws-best-lay { background: #f7b4cb; border-color: rgba(190,24,93,0.30); font-weight: 800; }
.ws-chip { color: #10233f; background: rgba(16,35,63,0.07); font-weight: 800; }
.ws-ev { color: #475569; background: rgba(148,163,184,0.14); font-weight: 800; }
.ws-ev-good { color: #0f766e; background: rgba(14,165,164,0.14); border: 1px solid rgba(14,165,164,0.26); }
.ws-ev-strong { color: #1d4ed8; background: rgba(59,130,246,0.16); border: 1px solid rgba(59,130,246,0.28); }
.ws-ev-neutral { color: #92400e; background: rgba(245,158,11,0.15); }
.ws-ev-bad { color: #991b1b; background: rgba(239,68,68,0.10); }
.ws-inner-section {
    margin: 8px 0 10px;
    padding: 6px 9px;
    border-left: 3px solid #0ea5a4;
    background: rgba(14,165,164,0.08);
    color: #10233f;
    font-weight: 800;
    font-size: 0.86rem;
    border-radius: 6px;
}
@media (max-width: 900px) {
    .ws-head { flex-direction: column; }
    .ws-right { justify-content: flex-start; }
    .ws-grid-head, .ws-grid-row { grid-template-columns: minmax(82px, 1.45fr) repeat(7, minmax(34px, 1fr)); gap: 1px; }
}
@media (max-width: 560px) {
    .ws-card { padding: 8px; }
    .ws-grid-head, .ws-grid-row { grid-template-columns: minmax(76px, 1.35fr) repeat(5, minmax(38px, 1fr)); }
    .ws-grid-head span:nth-child(2),
    .ws-grid-row > *:nth-child(2),
    .ws-grid-head span:nth-child(5),
    .ws-grid-row > *:nth-child(5) { display: none; }
    .ws-grid-row strong, .ws-chip, .ws-ev { min-height: 30px; }
    .ws-ids { font-size: 9px; }
}
@media (max-width: 420px) {
    .ws-grid-head, .ws-grid-row { grid-template-columns: minmax(68px, 1.25fr) repeat(5, minmax(32px, 1fr)); }
    .ws-grid-head span { font-size: 8px; }
    .ws-grid-row .p-col { font-size: 9px; }
}
</style>
"""


def _render_cards_grid(rows: pd.DataFrame, layout_choice: int, min_ev: float) -> None:
    columns = st.columns(int(layout_choice))
    for index, (_, row) in enumerate(rows.iterrows()):
        with columns[index % int(layout_choice)]:
            st.html(_market_card_html(row, min_ev))


def _render_inner_section(
    title: str, rows: pd.DataFrame, layout_choice: int, min_ev: float
) -> None:
    if rows.empty:
        return
    st.html(f"<div class='ws-inner-section'>{escape(str(title))} ({len(rows)})</div>")
    _render_cards_grid(rows, layout_choice, min_ev)


def render_upcoming_matches() -> None:
    db_status = get_db_status()
    if db_status["connected"] != "true":
        st.title("Matchs a venir")
        st.caption("Connexion FootNet requise pour charger WS_odds.")
        st.info(db_status["reason"])
        return

    st.title("Matchs a venir")
    st.caption(
        "Monitoring FootNet des matchs futurs WS_odds, adapte en 1X2 avec cotes back/lay et EV issues des probabilites calibrees."
    )

    with st.sidebar:
        st.divider()
        st.markdown("### Matchs a venir")
        layout_choice = st.selectbox(
            "Colonnes", [1, 2, 3], index=1, key="upcoming_layout"
        )
        sort_choice = st.radio(
            "Trier par",
            ["Date / heure de debut", "Ligue"],
            index=0,
            key="upcoming_sort_mode",
        )
        st.divider()
        if st.button(
            "Rafraichir", use_container_width=True, key="refresh_upcoming_matches"
        ):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()

    with st.spinner("Chargement matchs futurs..."):
        df = load_upcoming_ws_odds(st.session_state.get("selected_user_id"))

    if df is None or df.empty:
        st.info("Aucun match futur disponible dans les predictions.")
        return

    prepared = _prepare_matches(df)
    if prepared.empty:
        st.info("Aucun match a venir disponible.")
        return

    now_ts = pd.Timestamp.now()
    prepared = prepared[
        prepared["match_date"].notna() & prepared["match_date"].ge(now_ts)
    ].copy()
    if prepared.empty:
        st.info("Aucun match futur disponible dans les predictions.")
        return

    league_options = _sort_options(
        prepared["League"].dropna().astype(str).unique().tolist()
    )
    status_options = _sort_options(
        prepared["status"].dropna().astype(str).unique().tolist()
    )

    filter_controls = st.columns((2.8, 1.6, 1.8, 2.4))
    selected_leagues = filter_controls[0].multiselect(
        "Ligues",
        league_options,
        default=[],
        placeholder="Toutes les ligues",
        key="upcoming_leagues_filter_v2",
    )
    selected_status = filter_controls[1].multiselect(
        "Statut",
        status_options,
        default=[],
        placeholder="Tous",
        key="upcoming_status_filter_v2",
    )
    market_scope = filter_controls[2].segmented_control(
        "Marches",
        ["Tous", "Avec WS", "Pred seules"],
        default="Tous",
        key="upcoming_market_scope",
    )
    search_text = (
        filter_controls[3]
        .text_input(
            "Recherche",
            value="",
            placeholder="Equipe, ligue, match...",
        )
        .strip()
        .lower()
    )

    metric_filters = st.columns((1.4, 3.0))
    only_opps = metric_filters[0].toggle("Opportunites", value=False)
    min_ev = metric_filters[1].slider(
        "EV minimum (%)",
        min_value=-20.0,
        max_value=30.0,
        value=MIN_EV_DEFAULT,
        step=1.0,
    )

    view = prepared.copy()
    if selected_leagues:
        view = view[view["League"].isin(selected_leagues)]
    if selected_status:
        view = view[view["status"].isin(selected_status)]
    if market_scope == "Avec WS":
        view = view[view["has_ws_odds"]]
    elif market_scope == "Pred seules":
        view = view[~view["has_ws_odds"]]
    if only_opps:
        view = view[view["best_ev"].ge(min_ev)]
    if search_text:
        search_cols = ["match_label", "League", "ID_MATCH", "ID_MARKET"]
        mask = pd.Series(False, index=view.index)
        for column in search_cols:
            if column in view.columns:
                mask = mask | view[column].astype(str).str.lower().str.contains(
                    search_text, na=False
                )
        view = view[mask]

    if view["match_date"].notna().any():
        min_dt = view["match_date"].min().to_pydatetime()
        max_dt = view["match_date"].max().to_pydatetime()
        if min_dt < max_dt:
            date_range = st.slider(
                "Plage date / heure",
                min_value=min_dt,
                max_value=max_dt,
                value=(min_dt, max_dt),
                format="DD/MM/YYYY HH:mm",
                step=timedelta(minutes=30),
            )
            view = view[
                (view["match_date"] >= pd.to_datetime(date_range[0]))
                & (view["match_date"] <= pd.to_datetime(date_range[1]))
            ]

    if view.empty:
        st.info("Aucun match ne correspond aux filtres.")
        return

    opportunity_count = int(view["best_ev"].ge(min_ev).sum())
    inplay_count = int(view["is_inplay_effective"].sum())
    ws_count = int(view["has_ws_odds"].sum()) if "has_ws_odds" in view.columns else 0
    pred_only_count = int(len(view) - ws_count)
    last_update = view["updated_at"].max()
    mean_updates = float(
        pd.to_numeric(view.get("n_updates"), errors="coerce").fillna(0).mean()
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Marches", f"{len(view)}")
    k2.metric("Avec WS", f"{ws_count}")
    k3.metric("Pred seules", f"{pred_only_count}")
    k4.metric("Opportunites", f"{opportunity_count}")
    k5.metric("Derniere update", _fmt_ts(last_update, seconds=True))
    st.caption(f"Inplay: {inplay_count} | Moyenne updates WS: {mean_updates:.1f}")

    st.divider()
    st.markdown("### Matchs futurs")
    st.html(_WS_CSS)

    if sort_choice == "Ligue":
        cards_df = view.sort_values(
            ["League", "match_date", "match_label"],
            ascending=[True, True, True],
            na_position="last",
        )
        cards_df = cards_df.assign(
            date_group=cards_df["match_date"].map(_date_group_label)
        )
        for league in _sort_options(
            cards_df["League"].dropna().astype(str).unique().tolist()
        ):
            league_df = cards_df[cards_df["League"] == league]
            if league_df.empty:
                continue
            with st.expander(
                f"{league} ({len(league_df)})", expanded=len(cards_df) <= 12
            ):
                date_order = (
                    league_df.groupby("date_group", dropna=False)["match_date"]
                    .min()
                    .sort_values(na_position="last")
                    .index.tolist()
                )
                for date_group in date_order:
                    time_df = league_df[league_df["date_group"] == date_group]
                    _render_inner_section(
                        str(date_group), time_df, int(layout_choice), min_ev
                    )
    else:
        cards_df = view.sort_values(
            ["match_date", "League", "match_label"],
            ascending=[True, True, True],
            na_position="last",
        )
        cards_df = cards_df.assign(
            date_group=cards_df["match_date"].map(_date_group_label)
        )
        date_order = (
            cards_df.groupby("date_group", dropna=False)["match_date"]
            .min()
            .sort_values(na_position="last")
            .index.tolist()
        )
        for date_group in date_order:
            date_df = cards_df[cards_df["date_group"] == date_group]
            if date_df.empty:
                continue
            with st.expander(
                f"{date_group} ({len(date_df)})", expanded=len(date_order) <= 2
            ):
                for league in _sort_options(
                    date_df["League"].dropna().astype(str).unique().tolist()
                ):
                    league_df = date_df[date_df["League"] == league]
                    _render_inner_section(league, league_df, int(layout_choice), min_ev)

    st.divider()
    with st.expander("Tableau detaille", expanded=False):
        export_cols = [
            "updated_at",
            "match_date",
            "League",
            "status",
            "is_inplay_effective",
            "ID_MATCH",
            "ID_MARKET",
            "match_label",
            "best_home_back",
            "best_home_lay",
            "fair_home",
            "ev_home_back",
            "ev_home_lay",
            "best_draw_back",
            "best_draw_lay",
            "fair_draw",
            "ev_draw_back",
            "ev_draw_lay",
            "best_away_back",
            "best_away_lay",
            "fair_away",
            "ev_away_back",
            "ev_away_lay",
            "best_ev",
            "n_updates",
            "market_url",
        ]
        table = view[
            [column for column in export_cols if column in view.columns]
        ].copy()
        if sort_choice == "Ligue":
            table = table.sort_values(
                ["League", "match_date", "best_ev"],
                ascending=[True, True, False],
                na_position="last",
            )
        else:
            table = table.sort_values(
                ["match_date", "League", "best_ev"],
                ascending=[True, True, False],
                na_position="last",
            )
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "updated_at": st.column_config.DatetimeColumn(
                    "Maj WS", format="DD/MM/YYYY HH:mm:ss"
                ),
                "match_date": st.column_config.DatetimeColumn(
                    "Date match", format="DD/MM/YYYY HH:mm"
                ),
                "is_inplay_effective": st.column_config.CheckboxColumn("Inplay"),
                "market_url": st.column_config.LinkColumn(
                    "OrbitX", display_text="ouvrir"
                ),
                "best_ev": st.column_config.NumberColumn("Best EV", format="%+.1f%%"),
            },
        )
        st.download_button(
            "Exporter CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="footnet_ws_odds_matchs_a_venir.csv",
            mime="text/csv",
            key="export_upcoming_ws_odds",
        )
