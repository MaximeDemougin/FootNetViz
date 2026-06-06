from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from betting_data import get_db_status, load_hdp_simulation_frame
from ui import render_hero


def _hdp_line_type(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    quarter = int(round(abs(float(value)) * 4)) % 4
    mapping = {
        0: "Entiere (.00)",
        1: "Quart (.25)",
        2: "Demi (.50)",
        3: "Trois-quarts (.75)",
    }
    return mapping.get(quarter, "Unknown")


def _hdp_line_sign(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    if float(value) > 0:
        return "Positif"
    if float(value) < 0:
        return "Negatif"
    return "Zero"


def _effective_lay_odds(lay_odds: pd.Series) -> pd.Series:
    lay = pd.to_numeric(lay_odds, errors="coerce")
    valid = lay > 1.0
    effective = pd.Series(np.nan, index=lay.index, dtype=float)
    effective.loc[valid] = 1.0 + (0.97 / (lay.loc[valid] - 1.0))
    return effective


def _build_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for record in df.to_dict("records"):
        base = {
            "market_id": record.get("market_id"),
            "link_market_id": record.get("link_market_id"),
            "link_match_id": record.get("link_match_id"),
            "ID_MATCH": record.get("ID_MATCH"),
            "LeagueName": record.get("LeagueName"),
            "match_date": record.get("match_date"),
            "updated_at": record.get("updated_at"),
            "result": record.get("result"),
            "hdp_line": record.get("hdp_line"),
            "home_team": record.get("HomeTeam_clean") or record.get("home_name"),
            "away_team": record.get("AwayTeam_clean") or record.get("away_name"),
        }
        rows.append(
            {
                **base,
                "side": "home",
                "back_odds": record.get("home_best_back"),
                "lay_odds": record.get("home_best_lay"),
                "pred_odds": record.get("home_pred_odds"),
                "opp_pred_odds": record.get("away_pred_odds"),
                "ev_back_pct": record.get("ev_home_back_pct"),
                "ev_lay_pct": record.get("ev_home_lay_pct"),
                "profit_back_u": record.get("home_back_profit_u"),
                "profit_lay_u": record.get("home_lay_profit_u"),
                "outcome_back_u": record.get("home_back_outcome_u"),
                "outcome_lay_u": record.get("home_lay_outcome_u"),
                "opp_back_outcome_u": record.get("away_back_outcome_u"),
            }
        )
        rows.append(
            {
                **base,
                "side": "away",
                "back_odds": record.get("away_best_back"),
                "lay_odds": record.get("away_best_lay"),
                "pred_odds": record.get("away_pred_odds"),
                "opp_pred_odds": record.get("home_pred_odds"),
                "ev_back_pct": record.get("ev_away_back_pct"),
                "ev_lay_pct": record.get("ev_away_lay_pct"),
                "profit_back_u": record.get("away_back_profit_u"),
                "profit_lay_u": record.get("away_lay_profit_u"),
                "outcome_back_u": record.get("away_back_outcome_u"),
                "outcome_lay_u": record.get("away_lay_outcome_u"),
                "opp_back_outcome_u": record.get("home_back_outcome_u"),
            }
        )

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df

    for column in [
        "back_odds",
        "lay_odds",
        "pred_odds",
        "opp_pred_odds",
        "ev_back_pct",
        "ev_lay_pct",
        "profit_back_u",
        "profit_lay_u",
        "outcome_back_u",
        "outcome_lay_u",
        "opp_back_outcome_u",
        "hdp_line",
    ]:
        long_df[column] = pd.to_numeric(long_df[column], errors="coerce")

    long_df["match_date"] = pd.to_datetime(long_df["match_date"], errors="coerce")
    long_df["updated_at"] = pd.to_datetime(long_df["updated_at"], errors="coerce")
    long_df["league"] = long_df["LeagueName"].fillna("Unknown").astype(str)
    long_df["match"] = (
        long_df["home_team"].fillna("Home").astype(str)
        + " vs "
        + long_df["away_team"].fillna("Away").astype(str)
    )
    long_df["line_label"] = long_df["hdp_line"].map(
        lambda value: f"{float(value):+.2f}" if pd.notna(value) else "NA"
    )
    long_df["line_type"] = long_df["hdp_line"].apply(_hdp_line_type)
    long_df["line_sign"] = long_df["hdp_line"].apply(_hdp_line_sign)
    long_df["line_type_sign"] = long_df["line_sign"] + " | " + long_df["line_type"]
    long_df["has_score"] = (
        long_df["profit_back_u"].notna() | long_df["profit_lay_u"].notna()
    )
    return long_df


def _simulate(df: pd.DataFrame, position: str) -> pd.DataFrame:
    simulation = df.copy()
    if position == "lay":
        simulation["used_odds"] = _effective_lay_odds(simulation["lay_odds"])
        simulation["fair_used_odds"] = pd.to_numeric(
            simulation["opp_pred_odds"], errors="coerce"
        )
        simulation["ev_pct"] = (
            simulation["used_odds"] / simulation["fair_used_odds"] - 1.0
        ) * 100.0
        simulation["outcome_u"] = pd.to_numeric(
            simulation["opp_back_outcome_u"], errors="coerce"
        )
        simulation["profit_u"] = np.where(
            simulation["outcome_u"] > 0,
            simulation["outcome_u"] * (simulation["used_odds"] - 1.0),
            simulation["outcome_u"],
        )
    else:
        simulation["used_odds"] = pd.to_numeric(
            simulation["back_odds"], errors="coerce"
        )
        simulation["fair_used_odds"] = pd.to_numeric(
            simulation["pred_odds"], errors="coerce"
        )
        simulation["ev_pct"] = pd.to_numeric(simulation["ev_back_pct"], errors="coerce")
        simulation["profit_u"] = pd.to_numeric(
            simulation["profit_back_u"], errors="coerce"
        )
        simulation["outcome_u"] = pd.to_numeric(
            simulation["outcome_back_u"], errors="coerce"
        )

    simulation = simulation.dropna(
        subset=["used_odds", "fair_used_odds", "ev_pct", "profit_u"]
    ).copy()
    simulation = simulation.loc[simulation["used_odds"] > 1.0]
    simulation = simulation.sort_values(
        ["match_date", "updated_at"], na_position="last"
    )
    simulation["stake_u"] = 1.0
    simulation["expected_profit_u"] = (
        simulation["stake_u"] * simulation["ev_pct"] / 100.0
    )
    simulation["cum_profit_u"] = simulation["profit_u"].cumsum()
    simulation["cum_expected_u"] = simulation["expected_profit_u"].cumsum()
    return simulation


def render_hdp_analysis() -> None:
    db_status = get_db_status()
    if db_status["connected"] != "true":
        st.title("Analyse HDP")
        st.caption("Connexion FootNet requise pour analyser WS_odds_hdp.")
        st.info(db_status["reason"])
        return

    raw = load_hdp_simulation_frame()
    if raw.empty:
        st.title("Analyse HDP")
        st.caption("Aucune ligne HDP disponible dans WS_odds_hdp.")
        return

    long_df = _build_long_frame(raw)
    if long_df.empty:
        st.title("Analyse HDP")
        st.caption("Aucune ligne exploitable pour simulation HDP.")
        return

    render_hero(
        title="Analyse des potentiels HDP",
        subtitle="Simulation de strategies HDP basee sur WS_odds_hdp + score reel OddsPortal.",
        eyebrow="FootNetViz",
    )

    coverage_cols = st.columns(4)
    coverage_cols[0].metric("Lignes HDP", f"{int(raw.shape[0])}")
    coverage_cols[1].metric("Avec score", f"{int(long_df['has_score'].sum() / 2)}")
    coverage_cols[2].metric(
        "Matchs distincts", f"{int(long_df['link_match_id'].nunique())}"
    )
    coverage_cols[3].metric("Ligues", f"{int(long_df['league'].nunique())}")

    st.caption(
        "Jointure utilisee: WS_odds_hdp.MatchId -> AsianOdds_feeds.MatchId -> Oddsportal_data.ID_MATCH (score), puis attribution des probabilites via hdp_line."
    )

    leagues = sorted(long_df["league"].dropna().unique().tolist())
    line_types = sorted(long_df["line_type"].dropna().unique().tolist())
    line_signs = sorted(long_df["line_sign"].dropna().unique().tolist())
    min_date = long_df["match_date"].min()
    max_date = long_df["match_date"].max()

    with st.sidebar:
        st.divider()
        st.markdown("### Simulation HDP")
        side_choice = st.selectbox("Cote", ["Les deux", "Home", "Away"], index=0)
        position_choice = st.selectbox("Position", ["Back", "Lay"], index=0)
        group_choice = st.selectbox(
            "Grouper par",
            ["Type de ligne", "Signe de ligne", "Type + signe"],
            index=0,
        )
        ev_min = st.slider(
            "EV min (%)", min_value=-10.0, max_value=25.0, value=2.0, step=0.5
        )
        odds_range = st.slider(
            "Odds utilisees",
            min_value=1.01,
            max_value=15.0,
            value=(1.01, 5.0),
            step=0.01,
        )
        selected_leagues = st.multiselect("Ligues", leagues, default=leagues)
        selected_line_types = st.multiselect(
            "Type de ligne HDP", line_types, default=line_types
        )
        selected_line_signs = st.multiselect(
            "Signe ligne HDP", line_signs, default=line_signs
        )
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.date_input(
                "Periode",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )
        else:
            date_range = None

    filtered = long_df.copy()
    filtered = filtered.loc[filtered["league"].isin(selected_leagues)]
    filtered = filtered.loc[filtered["line_type"].isin(selected_line_types)]
    filtered = filtered.loc[filtered["line_sign"].isin(selected_line_signs)]
    if side_choice != "Les deux":
        filtered = filtered.loc[filtered["side"] == side_choice.lower()]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered.loc[
            filtered["match_date"].dt.date.between(start_date, end_date)
        ]

    simulation = _simulate(filtered, position_choice.lower())
    simulation = simulation.loc[simulation["ev_pct"] >= ev_min]
    simulation = simulation.loc[
        simulation["used_odds"].between(float(odds_range[0]), float(odds_range[1]))
    ]
    if simulation.empty:
        st.warning("Aucun scenario ne correspond aux filtres de simulation.")
        return

    st.caption(
        "Mode Lay: cote utilisee = 1 + 0.97 / (cote_lay - 1), puis pari sur le cote oppose."
    )

    bet_count = int(simulation.shape[0])
    profit_total = float(simulation["profit_u"].sum())
    roi_pct = (profit_total / bet_count) * 100.0 if bet_count else 0.0
    ev_total_u = float(simulation["expected_profit_u"].sum())

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Bets simules", str(bet_count))
    kpi_cols[1].metric("Profit (u)", f"{profit_total:+.2f}u")
    kpi_cols[2].metric("ROI / bet", f"{roi_pct:+.1f}%")
    kpi_cols[3].metric("EV totale (u)", f"{ev_total_u:+.2f}u")
    kpi_cols[4].metric("EV moyen", f"{float(simulation['ev_pct'].mean()):+.1f}%")

    axis_mode = st.selectbox(
        "Abscisse courbe",
        ["Index du bet", "Date"],
        index=0,
    )

    chart_df = simulation.copy()
    chart_df["point"] = chart_df["match_date"].fillna(chart_df["updated_at"])
    chart_df = chart_df.sort_values("point")
    chart_df["bet_index"] = np.arange(1, len(chart_df) + 1)
    chart_df["cum_profit_u"] = chart_df["profit_u"].cumsum()
    chart_df["cum_expected_u"] = chart_df["expected_profit_u"].cumsum()
    x_col = "bet_index" if axis_mode == "Index du bet" else "point"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df[x_col],
            y=chart_df["cum_profit_u"],
            mode="lines+markers",
            name="PnL cumule",
            line=dict(color="#0ea5a4", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df[x_col],
            y=chart_df["cum_expected_u"],
            mode="lines",
            name="EV cumulee",
            line=dict(color="#f59e0b", width=2.5, dash="dot"),
        )
    )
    fig.update_layout(title="Courbe cumulee: PnL reel vs EV (1u par bet)")
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    if axis_mode == "Date":
        fig.update_xaxes(type="date", tickformat="%d/%m %H:%M")
    else:
        fig.update_xaxes(title="Index du bet")
    st.plotly_chart(fig, use_container_width=True)

    group_col_map = {
        "Type de ligne": "line_type",
        "Signe de ligne": "line_sign",
        "Type + signe": "line_type_sign",
    }
    group_col = group_col_map[group_choice]

    grouped = (
        simulation.groupby(group_col, dropna=False)
        .agg(
            bets=("market_id", "size"),
            profit_u=("profit_u", "sum"),
            expected_u=("expected_profit_u", "sum"),
            ev_mean_pct=("ev_pct", "mean"),
            win_rate=("outcome_u", lambda series: (series > 0).mean() * 100.0),
        )
        .reset_index()
    )
    grouped["roi_pct"] = np.where(
        grouped["bets"] > 0,
        grouped["profit_u"] / grouped["bets"] * 100.0,
        0.0,
    )
    grouped = grouped.sort_values("profit_u", ascending=False)

    fig_group = px.bar(
        grouped,
        x=group_col,
        y=["profit_u", "expected_u"],
        barmode="group",
        title=f"Performance par {group_choice.lower()}",
    )
    fig_group.update_layout(
        height=340, margin=dict(l=20, r=20, t=50, b=20), legend_title_text="Mesure"
    )
    st.plotly_chart(fig_group, use_container_width=True)

    grouped_display = grouped.rename(
        columns={
            group_col: "Groupe HDP",
            "bets": "Bets",
            "profit_u": "Profit (u)",
            "expected_u": "EV attendue (u)",
            "ev_mean_pct": "EV moyenne %",
            "roi_pct": "ROI / bet %",
            "win_rate": "Win rate %",
        }
    )
    st.markdown(f"### Tableau par {group_choice.lower()}")
    st.dataframe(grouped_display, use_container_width=True)

    display = simulation[
        [
            "match_date",
            "league",
            "match",
            "side",
            "line_label",
            "line_type",
            "line_sign",
            "used_odds",
            "back_odds",
            "lay_odds",
            "fair_used_odds",
            "ev_pct",
            "expected_profit_u",
            "result",
            "profit_u",
            "outcome_u",
            "market_id",
            "link_market_id",
        ]
    ].copy()
    display = display.rename(
        columns={
            "match_date": "Date match",
            "league": "Ligue",
            "match": "Match",
            "side": "Side",
            "line_label": "HDP line",
            "line_type": "Type ligne",
            "line_sign": "Signe ligne",
            "used_odds": "Cote utilisee",
            "back_odds": "Back",
            "lay_odds": "Lay",
            "fair_used_odds": "Fair utilisee",
            "ev_pct": "EV %",
            "expected_profit_u": "EV (u)",
            "result": "Score",
            "profit_u": "Profit (u)",
            "outcome_u": "Outcome (u)",
            "market_id": "HDP market",
            "link_market_id": "1X2 market",
        }
    )
    st.dataframe(
        display.sort_values("Date match", ascending=False), use_container_width=True
    )
