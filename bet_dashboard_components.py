from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BIN_SORT_ORDERS = {
    "Cote": ["<1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-5.0", ">=5.0"],
    "Marge": ["<0%", "0-2%", "2-5%", "5-10%", "10-20%", ">=20%"],
}


def _fmt_money(value: float, unit_mode: bool = False) -> str:
    if unit_mode:
        return f"{value:.2f} u" if value >= 0 else f"{value:.2f} u"
    return (f"{value:,.2f}" if value >= 0 else f"{value:,.2f}").replace(",", " ")


def _fmt_signed_value(
    value: float,
    decimals: int = 2,
    suffix: str = "",
    unit_mode: bool = False,
    thousands: bool = True,
) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    format_spec = f",.{decimals}f" if thousands else f".{decimals}f"
    formatted = format(number, format_spec).replace(",", " ")
    return f"{formatted}{suffix}"


def _fmt_percent(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}%"


def _fmt_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        number = float(value)
    except Exception:
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.9f}".rstrip("0").rstrip(".")


def _gain_color(value: float) -> str:
    try:
        color = "#32b296" if float(value) > 0 else "#e04e4e"
        return f"color: {color}; font-weight: 700;"
    except Exception:
        return ""


def _neutral_color(_: object) -> str:
    return "color: #f59e0b; font-weight: 700;"


def _expected_color(value: float) -> str:
    try:
        color = "#3b82f6" if float(value) > 0 else "#e04e4e"
        return f"color: {color}; font-weight: 700;"
    except Exception:
        return ""


def _competition_color(value: object) -> str:
    color_map = {
        "match odds": "#10b981",
    }
    color = color_map.get(str(value).lower(), "#d1d4dc")
    return f"color: {color}; font-weight: 700;"


def _match_gain_style(row: pd.Series) -> list[str]:
    if str(row.get("Résultat", "")).lower() == "open":
        return ["", "color: #10233f; font-weight: 700;"]
    return ["", _gain_color(row.get("Gains net"))]


def _sort_grouped_rows(
    grouped: pd.DataFrame,
    group_by: str,
    group_col: str,
    sort_column: str,
    descending: bool,
) -> pd.DataFrame:
    natural_sort_columns = {group_by, group_col}
    if sort_column not in natural_sort_columns or group_by not in BIN_SORT_ORDERS:
        return grouped.sort_values(
            sort_column,
            ascending=not descending,
            na_position="last",
        )

    base_order = BIN_SORT_ORDERS[group_by]
    ordered_values = list(reversed(base_order)) if descending else list(base_order)
    sort_series = grouped[sort_column].astype(str)
    if sort_series.eq("Unknown").any():
        ordered_values = [*ordered_values, "Unknown"]

    sorted_grouped = grouped.copy()
    sorted_grouped[sort_column] = pd.Categorical(
        sort_series,
        categories=ordered_values,
        ordered=True,
    )
    return sorted_grouped.sort_values(sort_column, ascending=True, na_position="last")


def _build_insight_segments(df: pd.DataFrame, min_n: int = 12) -> pd.DataFrame:
    dimensions = [
        ("Résultat", "Résultat"),
        ("Etat offre", "Etat offre"),
        ("Ligue", "Ligue"),
        ("Cote", "Cote_bin"),
        ("Marge", "Marge_bin"),
    ]
    rows: list[pd.DataFrame] = []
    for label, column in dimensions:
        if column not in df.columns:
            continue
        segment = (
            df.dropna(subset=[column])
            .groupby(column, dropna=True)
            .agg(
                n=("Mise", "size"),
                mises=("Mise", "sum"),
                gains=("Gains net", "sum"),
                attendu=("Marge attendue", "sum"),
            )
            .reset_index()
        )
        if segment.empty:
            continue
        segment = segment.loc[segment["n"] >= min_n].copy()
        if segment.empty:
            continue
        segment["ROI"] = np.where(
            segment["mises"] > 0, segment["gains"] / segment["mises"] * 100, 0.0
        )
        segment["ROI_attendu"] = np.where(
            segment["mises"] > 0, segment["attendu"] / segment["mises"] * 100, 0.0
        )
        segment["Edge"] = segment["ROI"] - segment["ROI_attendu"]
        segment["dim"] = label
        segment = segment.rename(columns={column: "value"})
        rows.append(
            segment[
                ["dim", "value", "n", "mises", "gains", "ROI", "ROI_attendu", "Edge"]
            ]
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def render_insights_panel(df: pd.DataFrame, min_n: int = 12, top_k: int = 3) -> None:
    insights = _build_insight_segments(df, min_n=min_n)
    if insights.empty:
        st.info(
            f"Pas assez de paris pour generer des insights (>= {min_n} par segment)."
        )
        return
    best = insights.sort_values("ROI", ascending=False).head(top_k)
    worst = insights.sort_values("ROI", ascending=True).head(top_k)
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### Points forts")
        for row in best.itertuples(index=False):
            st.markdown(
                (
                    "<div class='footnet-panel'>"
                    f"<div style='font-size:0.76rem;color:#6b7280'>{row.dim} · n={int(row.n)}</div>"
                    f"<div style='font-weight:700'>{row.value}</div>"
                    f"<div style='color:#32b296;font-size:1.2rem;font-weight:700'>+{row.ROI:.1f}%</div>"
                    f"<div class='footnet-note'>Mises {row.mises:.2f} · Gains {row.gains:+.2f} · Edge {row.Edge:+.1f} pt</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
    with col_right:
        st.markdown("### A surveiller")
        for row in worst.itertuples(index=False):
            st.markdown(
                (
                    "<div class='footnet-panel'>"
                    f"<div style='font-size:0.76rem;color:#6b7280'>{row.dim} · n={int(row.n)}</div>"
                    f"<div style='font-weight:700'>{row.value}</div>"
                    f"<div style='color:#e04e4e;font-size:1.2rem;font-weight:700'>{row.ROI:.1f}%</div>"
                    f"<div class='footnet-note'>Mises {row.mises:.2f} · Gains {row.gains:+.2f} · Edge {row.Edge:+.1f} pt</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_distribution_charts(df: pd.DataFrame) -> None:
    if df.empty:
        return
    odds_distribution = df.groupby("Cote_bin", as_index=False).agg(
        gains=("Gains net", "sum"), mises=("Mise", "sum"), count=("Mise", "count")
    )
    margin_distribution = df.groupby("Marge_bin", as_index=False).agg(
        gains=("Gains net", "sum"), mises=("Mise", "sum"), count=("Mise", "count")
    )
    competition_distribution = df.groupby("Résultat", as_index=False).agg(
        gains=("Gains net", "sum"), count=("Mise", "count")
    )
    state_distribution = df.groupby("Etat offre", as_index=False).agg(
        gains=("Gains net", "sum"), count=("Mise", "count")
    )
    charts = st.columns(2)
    with charts[0]:
        fig = px.bar(
            odds_distribution,
            x="Cote_bin",
            y="gains",
            text="count",
            color="gains",
            color_continuous_scale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
            labels={"Cote_bin": "Cote", "gains": "Gains nets"},
        )
        fig.update_layout(
            title="Distribution par cote",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
        )
        st.plotly_chart(fig, width="stretch")
    with charts[1]:
        fig = px.bar(
            margin_distribution,
            x="Marge_bin",
            y="gains",
            text="count",
            color="gains",
            color_continuous_scale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#3b82f6"]],
            labels={"Marge_bin": "ROI attendu", "gains": "Gains nets"},
        )
        fig.update_layout(
            title="Distribution par ROI attendu",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
        )
        st.plotly_chart(fig, width="stretch")
    charts2 = st.columns(2)
    with charts2[0]:
        fig = px.bar(
            competition_distribution,
            x="Résultat",
            y="gains",
            text="count",
            color="Résultat",
            color_discrete_map={
                "Win": "#10b981",
                "Loss": "#ef4444",
                "Push": "#64748b",
                "Open": "#f59e0b",
            },
        )
        fig.update_layout(
            title="Performance par resultat",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
        )
        st.plotly_chart(fig, width="stretch")
    with charts2[1]:
        fig = px.bar(
            state_distribution,
            x="Etat offre",
            y="gains",
            text="count",
            color="Etat offre",
            color_discrete_map={
                "WON": "#10b981",
                "LOST": "#ef4444",
                "VOIDED": "#64748b",
            },
        )
        fig.update_layout(
            title="Performance par etat offre",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
        )
        st.plotly_chart(fig, width="stretch")


def render_cumulative_chart(
    bets_data: pd.DataFrame,
    mode: str = "match",
    unit_mode: bool = False,
    show_drawdown: bool = False,
    show_peaks: bool = True,
) -> None:
    if bets_data.empty:
        st.info("Aucune donnee a afficher")
        return

    plot_df: pd.DataFrame | None = None
    x_col = "Match_Num"
    normalized_mode = (mode or "match").lower()

    if normalized_mode == "jour" and "Date" in bets_data.columns:
        per_day = bets_data.copy()
        per_day["Date"] = pd.to_datetime(per_day["Date"], errors="coerce")
        per_day = per_day.dropna(subset=["Date"]).sort_values("Date")
        if not per_day.empty:
            per_day["Date_only"] = per_day["Date"].dt.normalize()
            plot_df = (
                per_day.groupby("Date_only", sort=True)
                .agg({"Gains net": "sum", "Marge attendue": "sum"})
                .reset_index()
                .rename(columns={"Date_only": "Date"})
            )
            plot_df["Cumulative Gains"] = plot_df["Gains net"].cumsum()
            plot_df["Cumulative_Marge"] = plot_df["Marge attendue"].cumsum()
            x_col = "Date"

    if plot_df is None:
        plot_df = bets_data.copy().reset_index(drop=True)
        plot_df["Match_Num"] = range(len(plot_df))
        if normalized_mode == "horaire" and "Date" in plot_df.columns:
            plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")
            plot_df = plot_df.sort_values("Date").reset_index(drop=True)
            plot_df["Match_Num"] = range(len(plot_df))
            x_col = "Date"

    plot_df = plot_df.copy()
    if "Cumulative Gains" not in plot_df.columns:
        plot_df["Cumulative Gains"] = plot_df.get(
            "Gains net", pd.Series([0.0] * len(plot_df))
        ).cumsum()
    if "Cumulative_Marge" not in plot_df.columns:
        plot_df["Cumulative_Marge"] = plot_df.get(
            "Marge attendue", pd.Series([0.0] * len(plot_df))
        ).cumsum()

    if unit_mode and "Mise" in plot_df.columns:
        stakes = pd.to_numeric(plot_df["Mise"], errors="coerce").replace(0, np.nan)
        plot_df["Cumulative Gains"] = (
            (pd.to_numeric(plot_df.get("Gains net", 0.0), errors="coerce") / stakes)
            .fillna(0.0)
            .cumsum()
        )
        plot_df["Cumulative_Marge"] = (
            (
                pd.to_numeric(plot_df.get("Marge attendue", 0.0), errors="coerce")
                / stakes
            )
            .fillna(0.0)
            .cumsum()
        )

    fmt = ".2f" if unit_mode else ".0f"
    y_suffix = " u" if unit_mode else "€"
    x_values = plot_df[x_col] if x_col in plot_df.columns else list(range(len(plot_df)))

    panel_bg = "rgba(255,255,255,0.82)"
    grid_color = "rgba(16,35,63,0.08)"
    axis_color = "#5e6d82"
    ink_color = "#10233f"
    expected_color = "#6b7c93"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=plot_df["Cumulative_Marge"],
            mode="lines",
            name="Attendu",
            line=dict(color=expected_color, width=1.4, dash="dash"),
            marker=dict(size=0),
            opacity=0.9,
            hovertemplate=f"%{{y:{fmt}}}{y_suffix}<extra>Attendu</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=plot_df["Cumulative Gains"],
            mode="lines+markers",
            name="Gains",
            line=dict(color="#0ea5a4", width=2.4),
            marker=dict(size=0, color="#0ea5a4", opacity=0),
            hovertemplate=f"%{{y:{fmt}}}{y_suffix}<extra>Gains</extra>",
        )
    )

    if show_peaks and len(plot_df) > 1:
        cum_series = pd.to_numeric(plot_df["Cumulative Gains"], errors="coerce")
        if cum_series.notna().any():
            imax = int(cum_series.idxmax())
            imin = int(cum_series.idxmin())
            peak_y = float(cum_series.loc[imax])
            trough_y = float(cum_series.loc[imin])
            peak_x = plot_df.iloc[imax][x_col]
            trough_x = plot_df.iloc[imin][x_col]
            fig.add_annotation(
                x=peak_x,
                y=peak_y,
                text=f"▲ Peak {_fmt_money(peak_y, unit_mode=unit_mode)}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#0ea5a4",
                ax=0,
                ay=-28,
                font=dict(color="#0b7c7b", size=11),
                bgcolor="rgba(255,255,255,0.96)",
                bordercolor="rgba(14,165,164,0.30)",
                borderwidth=1,
                borderpad=3,
            )
            if imin != imax:
                fig.add_annotation(
                    x=trough_x,
                    y=trough_y,
                    text=f"▼ Low {_fmt_money(trough_y, unit_mode=unit_mode)}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="#e04e4e",
                    ax=0,
                    ay=28,
                    font=dict(color="#e04e4e", size=11),
                    bgcolor="rgba(255,255,255,0.96)",
                    bordercolor="rgba(224,78,78,0.24)",
                    borderwidth=1,
                    borderpad=3,
                )

    if show_drawdown and len(plot_df) > 0:
        cum_vals = pd.to_numeric(plot_df["Cumulative Gains"], errors="coerce").fillna(
            0.0
        )
        running_max = np.maximum.accumulate(cum_vals.to_numpy())
        drawdown_vals = cum_vals.to_numpy() - running_max
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=drawdown_vals,
                mode="lines",
                name="Drawdown",
                line=dict(color="#e04e4e", width=1),
                fill="tozeroy",
                fillcolor="rgba(224,78,78,0.16)",
                hovertemplate=f"%{{y:{fmt}}}{y_suffix}<extra>Drawdown</extra>",
            )
        )

    fig.update_layout(
        plot_bgcolor=panel_bg,
        paper_bgcolor=panel_bg,
        font=dict(color=ink_color),
        hovermode="x unified",
        height=430,
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.98)",
            bordercolor="rgba(16,35,63,0.14)",
            font=dict(color=ink_color),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=axis_color),
            title_text="Légende",
        ),
        margin=dict(t=54, b=28, l=52, r=26),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color=axis_color, size=11),
        linecolor="rgba(255,255,255,0)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        zeroline=True,
        zerolinecolor="rgba(16,35,63,0.14)",
        tickfont=dict(color=axis_color, size=11),
        linecolor="rgba(16,35,63,0)",
    )
    if x_col == "Date":
        fig.update_xaxes(type="date", tickformat="%d %b %Y\n%H:%M")

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_match_card(df: pd.DataFrame, unit_mode: bool = False) -> None:
    if df.empty:
        st.info("Aucun match a afficher")
        return

    match = df.sort_values("Date", ascending=False).iloc[0]
    stake = float(pd.to_numeric(match.get("Mise"), errors="coerce") or 0.0)
    gains = float(pd.to_numeric(match.get("Gains net"), errors="coerce") or 0.0)
    expected = float(pd.to_numeric(match.get("Marge attendue"), errors="coerce") or 0.0)
    expected_roi = (expected / stake * 100) if stake > 0 else 0.0
    outcome_label = str(match.get("Résultat") or "Open")
    offer_state = str(match.get("Etat offre") or "Unknown")
    league = str(match.get("Ligue") or "-")
    selection = str(match.get("Modalité") or match.get("Selection") or "-")
    match_name = str(match.get("Match") or "-")
    match_date = pd.to_datetime(
        match.get("Date match", match.get("Date")), errors="coerce"
    )
    date_label = match_date.strftime("%Y-%m-%d %H:%M") if pd.notna(match_date) else "-"
    odds = pd.to_numeric(match.get("Cote"), errors="coerce")

    gain_color = "#0ea5a4" if gains >= 0 else "#e04e4e"
    outcome_bg = "rgba(14,165,164,0.10)" if gains >= 0 else "rgba(224,78,78,0.10)"
    expected_fill = max(0.0, min(100.0, expected_roi * 5 if expected_roi > 0 else 0.0))
    expected_bar = (
        "linear-gradient(90deg,#0ea5a4,#22c55e)"
        if expected_roi > 0
        else "linear-gradient(90deg,#f59e0b,#f97316)"
    )

    def _fmt_value(value: float, suffix: str) -> str:
        if unit_mode:
            return _fmt_signed_value(value, decimals=2, suffix=" u", unit_mode=True)
        return _fmt_signed_value(value, decimals=2, suffix=suffix)

    stake_label = "1.00 u" if unit_mode else f"{stake:,.2f}€".replace(",", " ")
    gains_label = _fmt_value(gains, "€")
    expected_label = _fmt_value(expected, "€")

    st.markdown("### Fiche match")
    st.markdown(
        (
            "<div class='footnet-panel' style='padding:1.15rem 1.2rem;'>"
            "<div style='display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;'>"
            "<div style='min-width:0;'>"
            f"<div style='font-size:1.05rem;font-weight:700;color:#10233f;line-height:1.25'>{match_name}</div>"
            f"<div style='margin-top:0.35rem;color:#5e6d82;font-size:0.92rem'>{league}</div>"
            "</div>"
            f"<div style='padding:0.55rem 0.8rem;border-radius:999px;background:{outcome_bg};text-align:center;min-width:112px;'>"
            "<div style='font-size:0.72rem;color:#5e6d82;margin-bottom:0.15rem'>Résultat</div>"
            f"<div style='font-size:1.05rem;font-weight:800;color:{gain_color};'>{gains_label}</div>"
            f"<div style='margin-top:0.2rem;font-size:0.72rem;color:#5e6d82'>{outcome_label}</div>"
            "</div>"
            "</div>"
            "<div style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0.7rem;margin-top:1rem;'>"
            f"<div style='background:rgba(14,165,164,0.06);border-radius:12px;padding:0.75rem;'><div style='font-size:0.72rem;color:#5e6d82'>Date</div><div style='font-weight:600;color:#10233f'>{date_label}</div></div>"
            f"<div style='background:rgba(14,165,164,0.06);border-radius:12px;padding:0.75rem;'><div style='font-size:0.72rem;color:#5e6d82'>Modalité</div><div style='font-weight:600;color:#10233f'>{selection}</div></div>"
            f"<div style='background:rgba(14,165,164,0.06);border-radius:12px;padding:0.75rem;'><div style='font-size:0.72rem;color:#5e6d82'>Mise</div><div style='font-weight:600;color:#10233f'>{stake_label}</div></div>"
            f"<div style='background:rgba(14,165,164,0.06);border-radius:12px;padding:0.75rem;'><div style='font-size:0.72rem;color:#5e6d82'>Cote</div><div style='font-weight:600;color:#10233f'>{'' if pd.isna(odds) else f'{float(odds):.3f}'}</div></div>"
            f"<div style='background:rgba(14,165,164,0.06);border-radius:12px;padding:0.75rem;'><div style='font-size:0.72rem;color:#5e6d82'>Etat offre</div><div style='font-weight:600;color:#10233f'>{offer_state}</div></div>"
            f"<div style='background:rgba(14,165,164,0.06);border-radius:12px;padding:0.75rem;'><div style='font-size:0.72rem;color:#5e6d82'>Marge attendue</div><div style='font-weight:600;color:#10233f'>{expected_label}</div></div>"
            "</div>"
            "<div style='margin-top:1rem;'>"
            "<div style='display:flex;justify-content:space-between;gap:0.75rem;font-size:0.78rem;color:#5e6d82;margin-bottom:0.35rem;'>"
            "<span>ROI attendu</span>"
            f"<span>{_fmt_percent(expected_roi, decimals=1)}</span>"
            "</div>"
            "<div style='height:10px;background:rgba(16,35,63,0.08);border-radius:999px;overflow:hidden;'>"
            f"<div style='width:{expected_fill:.1f}%;height:100%;background:{expected_bar};'></div>"
            "</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_grouped_charts(
    display_df: pd.DataFrame,
    group_by: str,
    unit_mode: bool = False,
) -> None:
    if display_df.empty or group_by == "Match":
        return

    chart_df = display_df.copy()
    chart_df[group_by] = chart_df[group_by].astype("string").fillna("Unknown")

    max_bars = (
        24 if group_by in {"Jour", "Semaine", "Jour de la semaine", "Mois"} else 16
    )
    if len(chart_df) > max_bars:
        chart_df = chart_df.head(max_bars).copy()
        st.caption(
            f"Visualisations limitées aux {max_bars} premières modalités affichées dans le tableau."
        )

    categories = chart_df[group_by].tolist()
    mises_label = "Mises (u)" if unit_mode else "Mises (€)"
    perf_label = "Gains (u)" if unit_mode else "Gains (€)"
    attendu_label = "Attendu (u)" if unit_mode else "Attendu (€)"
    value_suffix = " u" if unit_mode else "€"
    value_format = ".2f" if unit_mode else ".0f"

    charts = st.columns(2)

    mises_fig = go.Figure()
    mises_fig.add_trace(
        go.Bar(
            x=chart_df[group_by],
            y=chart_df["Mises"],
            marker=dict(color="#9aa4b2"),
            text=chart_df.apply(
                lambda row: (
                    f"{int(row['Nb Paris'])}p\n{row['Mises']:.2f}u"
                    if unit_mode
                    else f"{int(row['Nb Paris'])}p\n{row['Mises']:,.0f}€".replace(
                        ",", " "
                    )
                ),
                axis=1,
            ),
            textposition="inside",
            textfont=dict(color="#ffffff", size=11),
            hovertemplate=f"%{{x}}<br>%{{y:{value_format}}}{value_suffix}<br>Nb Paris: %{{customdata}}<extra>{mises_label}</extra>",
            customdata=chart_df["Nb Paris"],
        )
    )
    mises_fig.update_layout(
        title=f"Mises par {group_by.lower()}",
        plot_bgcolor="rgba(255,255,255,0.82)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#10233f"),
        margin=dict(t=44, b=24, l=18, r=18),
        height=360,
        xaxis=dict(categoryorder="array", categoryarray=categories, title=None),
        yaxis=dict(title=None),
    )
    mises_fig.update_xaxes(tickangle=-30, showgrid=False)
    mises_fig.update_yaxes(gridcolor="rgba(16,35,63,0.08)")

    perf_fig = go.Figure()
    perf_fig.add_trace(
        go.Bar(
            x=chart_df[group_by],
            y=chart_df["Gains"],
            marker=dict(color=np.where(chart_df["Gains"] >= 0, "#32b296", "#e04e4e")),
            text=chart_df["Gains"].map(
                lambda value: (
                    _fmt_signed_value(value, decimals=2, suffix="u", thousands=False)
                    if unit_mode
                    else _fmt_signed_value(value, decimals=0, suffix="€")
                )
            ),
            textposition="outside",
            hovertemplate=f"%{{x}}<br>%{{y:{value_format}}}{value_suffix}<extra>{perf_label}</extra>",
            name="Gains",
        )
    )
    if "Attendu" in chart_df.columns:
        perf_fig.add_trace(
            go.Scatter(
                x=chart_df[group_by],
                y=chart_df["Attendu"],
                mode="lines+markers",
                name="Attendu",
                line=dict(color="#3b82f6", width=2, dash="dash"),
                marker=dict(size=7, color="#3b82f6"),
                hovertemplate=f"%{{x}}<br>%{{y:{value_format}}}{value_suffix}<extra>{attendu_label}</extra>",
            )
        )
    perf_fig.update_layout(
        title=f"Performance par {group_by.lower()}",
        plot_bgcolor="rgba(255,255,255,0.82)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#10233f"),
        margin=dict(t=44, b=24, l=18, r=18),
        height=360,
        barmode="relative",
        xaxis=dict(categoryorder="array", categoryarray=categories, title=None),
        yaxis=dict(title=None),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    perf_fig.update_xaxes(tickangle=-30, showgrid=False)
    perf_fig.update_yaxes(
        gridcolor="rgba(16,35,63,0.08)", zerolinecolor="rgba(16,35,63,0.14)"
    )

    with charts[0]:
        st.plotly_chart(mises_fig, width="stretch", config={"displayModeBar": False})
    with charts[1]:
        st.plotly_chart(perf_fig, width="stretch", config={"displayModeBar": False})


def render_grouped_table(df: pd.DataFrame, unit_mode: bool = False) -> None:
    if df.empty:
        st.info("Aucune donnee a afficher")
        return

    group_by = st.radio(
        "Grouper par",
        options=[
            "Match",
            "Pays",
            "Ligue",
            "Cote",
            "Marge",
            "Mois",
            "Semaine",
            "Jour",
            "Jour de la semaine",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )
    col_map = {
        "Match": "Match",
        "Pays": "Pays",
        "Ligue": "Ligue",
        "Cote": "Cote_bin",
        "Marge": "Marge_bin",
        "Mois": "Mois",
        "Semaine": "Semaine",
        "Jour": "Jour",
        "Jour de la semaine": "Jour de la semaine",
    }
    group_col = col_map[group_by]
    sort_key_map = {
        "Mois": "Mois_key",
        "Semaine": "Semaine_key",
        "Jour": "Jour",
        "Jour de la semaine": "Jour de la semaine_key",
    }

    if group_by == "Match":
        match_columns = [
            "MatchId",
            "ID_MARKET",
            "Date",
            "Ligue",
            "Match",
            "Modalité",
            "Résultat",
            "Cote",
            "Prédiction",
            "Mise",
            "Marge attendue",
            "ROI attendu %",
            "Gains net",
        ]
        display_match = df[match_columns].copy().rename(
            columns={"MatchId": "ID AsianOdds", "ID_MARKET": "ID Market"}
        )
        if "Date match" in df.columns:
            display_match["Date"] = df["Date match"]
        display_match = display_match.sort_values("Date", ascending=False)
        if unit_mode:
            display_match["Gains net"] = display_match["Gains net"] / display_match[
                "Mise"
            ].replace(0, np.nan)
            display_match["Marge attendue"] = display_match[
                "Marge attendue"
            ] / display_match["Mise"].replace(0, np.nan)
            display_match["Mise"] = 1.0
        styler = display_match.style.format(
            {
                "ID AsianOdds": _fmt_identifier,
                "ID Market": _fmt_identifier,
                "Date": lambda value: (
                    pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")
                    if pd.notna(value)
                    else ""
                ),
                "Cote": lambda value: f"{value:.3f}" if pd.notna(value) else "",
                "Prédiction": lambda value: f"{value:.3f}" if pd.notna(value) else "",
                "Mise": (lambda value: f"{value:.0f} u")
                if unit_mode
                else (lambda value: f"{value:,.2f}€".replace(",", " ")),
                "Gains net": (
                    lambda value: _fmt_signed_value(
                        value, decimals=2, suffix=" u", unit_mode=True, thousands=False
                    )
                )
                if unit_mode
                else (lambda value: _fmt_signed_value(value, decimals=2, suffix="€")),
                "Marge attendue": (
                    lambda value: _fmt_signed_value(
                        value, decimals=2, suffix=" u", unit_mode=True, thousands=False
                    )
                )
                if unit_mode
                else (lambda value: _fmt_signed_value(value, decimals=2, suffix="€")),
                "ROI attendu %": lambda value: _fmt_percent(value, decimals=1),
            }
        )
        styler = styler.map(_gain_color, subset=["ROI attendu %"])
        styler = styler.apply(
            _match_gain_style,
            axis=1,
            subset=["Résultat", "Gains net"],
        )
        styler = styler.map(_expected_color, subset=["Marge attendue"])
        styler = styler.map(_neutral_color, subset=["Mise"])
        st.dataframe(styler, width="stretch", hide_index=True)
        return

    aggregations: dict[str, tuple[str, str]] = {
        "Count": ("Mise", "count"),
        "Avg_Cote": ("Cote", "mean"),
        "Total_Mises": ("Mise", "sum"),
        "Total_Gains": ("Gains net", "sum"),
        "Resultat_attendu": ("Marge attendue", "sum"),
    }
    if group_by in sort_key_map and sort_key_map[group_by] in df.columns:
        aggregations["Sort_key"] = (sort_key_map[group_by], "first")

    grouped = df.groupby(group_col, dropna=False).agg(**aggregations).reset_index()
    grouped["ROI"] = np.where(
        grouped["Total_Mises"] > 0,
        grouped["Total_Gains"] / grouped["Total_Mises"] * 100,
        0.0,
    )
    grouped["ROI_attendu"] = np.where(
        grouped["Total_Mises"] > 0,
        grouped["Resultat_attendu"] / grouped["Total_Mises"] * 100,
        0.0,
    )

    if group_by in {"Mois", "Semaine", "Jour"} and "Sort_key" in grouped.columns:
        grouped = grouped.sort_values("Sort_key", ascending=False, na_position="last")
    elif group_by == "Jour de la semaine" and "Sort_key" in grouped.columns:
        grouped = grouped.sort_values("Sort_key", ascending=True, na_position="last")
    else:
        sort_columns = {
            group_by: group_col,
            "Gains": "Total_Gains",
            "ROI": "ROI",
            "Mises": "Total_Mises",
            "Nb Paris": "Count",
            "Attendu": "Resultat_attendu",
            "ROI attendu": "ROI_attendu",
        }
        default_sort_label = (
            group_by if group_by in {"Pays", "Ligue", "Cote", "Marge"} else "Gains"
        )
        descending_default_map = {
            "Pays": False,
            "Ligue": False,
            "Cote": False,
            "Marge": True,
        }
        sort_controls = st.columns((1.15, 0.85))
        sort_label = sort_controls[0].selectbox(
            "Trier par",
            options=list(sort_columns.keys()),
            index=list(sort_columns.keys()).index(default_sort_label),
            key=f"group_sort_{group_by}",
        )
        descending_default = descending_default_map.get(
            group_by, sort_label != group_by
        )
        descending = sort_controls[1].toggle(
            "Décroissant",
            value=descending_default,
            key=f"group_sort_desc_{group_by}",
        )
        grouped = _sort_grouped_rows(
            grouped,
            group_by=group_by,
            group_col=group_col,
            sort_column=sort_columns[sort_label],
            descending=descending,
        )

    display_df = grouped.rename(
        columns={
            group_col: group_by,
            "Count": "Nb Paris",
            "Avg_Cote": "Cote moyenne",
            "Total_Mises": "Mises",
            "Total_Gains": "Gains",
            "ROI": "ROI",
            "Resultat_attendu": "Attendu",
            "ROI_attendu": "ROI attendu",
        }
    )
    if unit_mode:
        display_df["Mises"] = display_df["Nb Paris"].astype(float)
        display_df["Gains"] = display_df["Gains"] / display_df["Nb Paris"].replace(
            0, np.nan
        )
        display_df["Attendu"] = display_df["Attendu"] / display_df["Nb Paris"].replace(
            0, np.nan
        )
    display_df = display_df.drop(columns=["Sort_key"], errors="ignore")
    preferred_order = [
        group_by,
        "Nb Paris",
        "Mises",
        "Cote moyenne",
        "Attendu",
        "ROI attendu",
        "Gains",
        "ROI",
    ]
    display_df = display_df[
        [column for column in preferred_order if column in display_df.columns]
    ]
    styler = display_df.style.format(
        {
            "Cote moyenne": lambda value: f"{value:.2f}" if pd.notna(value) else "",
            "Mises": (lambda value: f"{value:,.0f} u".replace(",", " "))
            if unit_mode
            else (lambda value: f"{value:,.0f}€".replace(",", " ")),
            "Gains": (
                lambda value: _fmt_signed_value(
                    value, decimals=2, suffix=" u", unit_mode=True, thousands=False
                )
            )
            if unit_mode
            else (lambda value: _fmt_signed_value(value, decimals=0, suffix="€")),
            "ROI": lambda value: _fmt_percent(value, decimals=1),
            "Attendu": (
                lambda value: _fmt_signed_value(
                    value, decimals=2, suffix=" u", unit_mode=True, thousands=False
                )
            )
            if unit_mode
            else (lambda value: _fmt_signed_value(value, decimals=0, suffix="€")),
            "ROI attendu": lambda value: _fmt_percent(value, decimals=1),
        }
    )
    styler = styler.map(_gain_color, subset=["Gains", "ROI", "ROI attendu"])
    styler = styler.map(_expected_color, subset=["Attendu"])
    styler = styler.map(_neutral_color, subset=["Mises"])
    if "Pays" in display_df.columns:
        styler = styler.map(_competition_color, subset=["Pays"])
    st.dataframe(styler, width="stretch", hide_index=True)
    _render_grouped_charts(display_df, group_by=group_by, unit_mode=unit_mode)
