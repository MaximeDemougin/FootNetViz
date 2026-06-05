import streamlit as st
import pandas as pd
import numpy as np

from bet_dashboard_components import (
    render_cumulative_chart,
    render_grouped_table,
    render_match_card,
)
from betting_data import (
    compute_dashboard_kpis,
    get_db_status,
    load_bet_results,
    prepare_dashboard_bets,
)


def render_dashboard() -> None:
    db_status = get_db_status()
    if db_status["connected"] != "true":
        st.title("Resultats FootNet")
        st.caption("Connexion FootNet requise pour charger les resultats utilisateur.")
        st.info(db_status["reason"])
        return

    user_id = st.session_state.get("selected_user_id")
    if user_id is None:
        st.title("Resultats FootNet")
        st.caption("Aucun utilisateur disponible.")
        return

    bets = load_bet_results(int(user_id))
    if bets.empty:
        st.title("Resultats FootNet")
        st.caption(f"Aucun resultat retourne pour l'utilisateur {user_id}.")
        return

    dashboard_bets = prepare_dashboard_bets(bets)

    date_min = dashboard_bets["Date"].min()
    date_max = dashboard_bets["Date"].max()
    odds_min = float(dashboard_bets["Cote"].min())
    odds_max = float(dashboard_bets["Cote"].max())
    roi_expected = (
        dashboard_bets["ROI attendu %"].replace([np.inf, -np.inf], pd.NA).dropna()
    )
    roi_min = float(roi_expected.min()) if not roi_expected.empty else -10.0
    roi_max = float(roi_expected.max()) if not roi_expected.empty else 50.0

    st.title("Resultats FootNet")
    st.caption(f"Historique utilisateur - {bets.iloc[0]['username']}")

    with st.sidebar:
        st.divider()
        st.markdown("### Dashboard")
        unit_mode = st.toggle("Unites", value=False, key="dashboard_unit_mode")
        st.divider()
        if st.button("Rafraichir", use_container_width=True, key="refresh_dashboard"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()

    lower_bar = st.columns((1.25, 1.0, 1.0))
    date_range = lower_bar[0].date_input(
        "Periode",
        value=(date_min.date(), date_max.date()),
        min_value=date_min.date(),
        max_value=date_max.date(),
    )
    odds_range = lower_bar[1].slider(
        "Cote",
        min_value=float(odds_min),
        max_value=float(odds_max),
        value=(float(odds_min), float(odds_max)),
        step=0.01,
    )
    roi_range = lower_bar[2].slider(
        "ROI attendu (%)",
        min_value=float(min(-10.0, roi_min)),
        max_value=float(max(roi_max, 20.0)),
        value=(float(min(-10.0, roi_min)), float(max(roi_max, 20.0))),
        step=0.5,
    )

    filtered = dashboard_bets.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered.loc[filtered["Date"].dt.date.between(start_date, end_date)]
    filtered = filtered.loc[
        filtered["Cote"].between(
            float(odds_range[0]), float(odds_range[1]), inclusive="both"
        )
    ]
    filtered = filtered.loc[
        filtered["ROI attendu %"]
        .fillna(0)
        .between(float(roi_range[0]), float(roi_range[1]), inclusive="both")
    ]

    if filtered.empty:
        st.warning("Aucun resultat ne correspond aux filtres courants.")
        return

    metric_source = filtered.assign(
        settled=filtered["Résultat"].ne("Open"),
        profit=filtered["Gains net"],
        stake=filtered["Mise"],
        expected_profit=filtered["Marge attendue"],
        day_label=filtered["Date"].dt.date,
        ID_BET=filtered["ID_BET"],
        match_type=filtered["Ligue"],
    )
    kpis = compute_dashboard_kpis(metric_source)
    metric_columns = st.columns(5)
    metric_columns[0].metric("PnL realise", f"{kpis['total_profit']:+.2f}")
    metric_columns[1].metric("ROI realise", f"{kpis['roi_pct']:+.1f}%")
    metric_columns[2].metric("Bets settles", str(kpis["settled_bets"]))
    metric_columns[3].metric("EV total", f"{kpis['expected_profit_total']:+.2f}")
    metric_columns[4].metric("EV %", f"{kpis['expected_roi_pct']:+.1f}%")

    chart_bets = filtered.sort_values("Date").copy()
    chart_bets["Cumulative Gains"] = chart_bets["Gains net"].cumsum()

    upper_row = st.columns((1.4, 1.0))
    with upper_row[0]:
        st.markdown("### Evolution des gains nets")
        chart_controls = st.columns((1.3, 0.75, 0.95))
        view_mode = chart_controls[0].selectbox(
            "Afficher",
            ["Par match", "Par horaire", "Par jour"],
            index=0,
            label_visibility="collapsed",
        )
        show_peaks = chart_controls[1].toggle("Peaks", value=True)
        show_drawdown = chart_controls[2].toggle("Drawdown", value=False)
        render_cumulative_chart(
            chart_bets,
            mode={
                "Par match": "match",
                "Par horaire": "horaire",
                "Par jour": "jour",
            }[view_mode],
            unit_mode=unit_mode,
            show_drawdown=show_drawdown,
            show_peaks=show_peaks,
        )
    with upper_row[1]:
        render_match_card(filtered, unit_mode=unit_mode)

    render_grouped_table(filtered, unit_mode=unit_mode)
