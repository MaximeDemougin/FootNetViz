from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

DEFAULT_DB_NAME = "FootNet"
ORBITX_COMMISSION_RATE = 0.03
EXCHANGE_COMMISSION_RATE = 0.03
MATCH_PRECISION = 6


def _get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    if value is None:
        env_name = f"FOOTNET_{name.upper()}"
        value = os.getenv(env_name)
    return str(value).strip() if value else None


def get_db_config() -> dict[str, str | bool]:
    db_url = _get_secret("db_url")
    db_name = _get_secret("db_name") or DEFAULT_DB_NAME
    return {
        "configured": bool(db_url),
        "db_url": db_url or "",
        "db_name": db_name,
    }


def _resolved_db_url() -> str:
    config = get_db_config()
    if not config["configured"]:
        raise RuntimeError("Le secret db_url n'est pas configure.")
    url = make_url(str(config["db_url"]))
    if not url.database:
        url = url.set(database=str(config["db_name"]))
    return url.render_as_string(hide_password=False)


@st.cache_resource(show_spinner=False)
def _get_engine(resolved_url: str) -> Engine:
    return create_engine(resolved_url, pool_pre_ping=True)


def _query_dataframe(query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    engine = _get_engine(_resolved_db_url())
    with engine.connect() as connection:
        return pd.read_sql_query(text(query), connection, params=params)


@st.cache_data(ttl=120, show_spinner=False)
def get_db_status() -> dict[str, str]:
    config = get_db_config()
    if not config["configured"]:
        return {
            "connected": "false",
            "reason": "Ajoutez db_url dans .streamlit/secrets.toml ou FOOTNET_DB_URL dans l'environnement.",
            "db_name": str(config["db_name"]),
        }
    try:
        heartbeat = _query_dataframe(
            "SELECT DATABASE() AS db_name, NOW() AS server_time"
        )
        return {
            "connected": "true",
            "reason": "",
            "db_name": str(heartbeat.loc[0, "db_name"]),
        }
    except Exception as exc:
        return {
            "connected": "false",
            "reason": str(exc),
            "db_name": str(config["db_name"]),
        }


@st.cache_data(ttl=120, show_spinner=False)
def get_user_catalog() -> pd.DataFrame:
    query = f"""
        WITH bp_ranked AS (
            SELECT
                bp.ID_BET,
                bp.ID_USER,
                bp.ID_MARKET,
                bp.odds,
                bp.stake,
                bp.created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        bp.ID_USER,
                        bp.ID_MARKET,
                        ROUND(COALESCE(bp.odds, 0), {MATCH_PRECISION}),
                        ROUND(COALESCE(bp.stake, 0), {MATCH_PRECISION})
                    ORDER BY bp.created_at, bp.ID_BET
                ) AS orbitx_match_rank
            FROM Bet_p bp
        ),
        ob_base AS (
            SELECT
                ob.ID_USER,
                ob.ID_MARKET,
                ob.profit AS raw_profit,
                ob.settledDate,
                ob.matchedDate,
                ob.price AS matched_odds,
                ob.size AS matched_stake,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        ob.ID_USER,
                        ob.ID_MARKET,
                        ROUND(COALESCE(ob.price, 0), {MATCH_PRECISION}),
                        ROUND(COALESCE(ob.size, 0), {MATCH_PRECISION})
                    ORDER BY COALESCE(ob.matchedDate, ob.placedDate, ob.settledDate), ob.id
                ) AS orbitx_match_rank
            FROM Orbitx_bets ob
        ),
        ob_scored AS (
            SELECT
                ob_base.*,
                SUM(ob_base.raw_profit) OVER (
                    PARTITION BY ob_base.ID_USER, ob_base.ID_MARKET
                ) AS market_raw_profit,
                SUM(
                    CASE WHEN ob_base.raw_profit > 0 THEN ob_base.raw_profit ELSE 0 END
                ) OVER (
                    PARTITION BY ob_base.ID_USER, ob_base.ID_MARKET
                ) AS market_positive_profit
            FROM ob_base
        ),
        ob_ranked AS (
            SELECT
                ob_scored.ID_USER,
                ob_scored.ID_MARKET,
                CASE
                    WHEN ob_scored.raw_profit > 0
                     AND ob_scored.market_raw_profit > 0
                     AND ob_scored.market_positive_profit > 0
                    THEN ob_scored.raw_profit - (
                        ob_scored.market_raw_profit * {ORBITX_COMMISSION_RATE:.2f}
                    ) * (ob_scored.raw_profit / ob_scored.market_positive_profit)
                    ELSE ob_scored.raw_profit
                END AS profit,
                ob_scored.settledDate,
                ob_scored.matchedDate,
                ob_scored.matched_odds,
                ob_scored.matched_stake,
                ob_scored.orbitx_match_rank
            FROM ob_scored
        )
        SELECT
            bp.ID_USER,
            COALESCE(u.username, CONCAT('User ', bp.ID_USER)) AS username,
            COUNT(*) AS total_bets,
            COALESCE(SUM(ob.profit), 0) AS total_profit,
            COALESCE(SUM(bp.stake), 0) AS total_stake,
            MAX(COALESCE(ob.settledDate, ob.matchedDate, bp.created_at)) AS last_activity
        FROM bp_ranked bp
        LEFT JOIN Users u
            ON bp.ID_USER = u.ID_USER
        LEFT JOIN ob_ranked ob
            ON bp.ID_USER = ob.ID_USER
           AND bp.ID_MARKET = ob.ID_MARKET
           AND ROUND(COALESCE(bp.odds, 0), {MATCH_PRECISION}) = ROUND(COALESCE(ob.matched_odds, 0), {MATCH_PRECISION})
           AND ROUND(COALESCE(bp.stake, 0), {MATCH_PRECISION}) = ROUND(COALESCE(ob.matched_stake, 0), {MATCH_PRECISION})
           AND bp.orbitx_match_rank = ob.orbitx_match_rank
        GROUP BY bp.ID_USER, COALESCE(u.username, CONCAT('User ', bp.ID_USER))
        ORDER BY total_profit DESC, total_bets DESC, username ASC
    """
    users = _query_dataframe(query)
    if users.empty:
        return users
    users["last_activity"] = pd.to_datetime(users["last_activity"], errors="coerce")
    users["label"] = users.apply(
        lambda row: (
            f"{row['username']} (ID {int(row['ID_USER'])})"
            f" · {int(row['total_bets'])} bets"
            f" · PnL {float(row['total_profit']):+.2f}"
        ),
        axis=1,
    )
    return users


def _aggregate_duplicate_bets(bets: pd.DataFrame) -> pd.DataFrame:
    """Merge multiple bets on the same match/market/selection into one line.

    Stakes, liabilities, expected gains and matched stakes are summed, while the
    placed odds (and matched odds) are stake-weighted averages of the legs.
    """
    if bets.empty:
        return bets

    group_keys = [
        key
        for key in (
            "ID_USER",
            "MatchId",
            "ID_MARKET",
            "bet",
            "bet_libelle",
            "side_back_lay",
        )
        if key in bets.columns
    ]
    if not group_keys or not bets.duplicated(subset=group_keys, keep=False).any():
        return bets

    df = bets.copy()
    stake = pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)
    odds = pd.to_numeric(df["placed_odds"], errors="coerce")
    matched_stake = pd.to_numeric(df.get("matched_stake"), errors="coerce")
    matched_odds = pd.to_numeric(df.get("matched_odds"), errors="coerce")

    df["_stake_w"] = stake
    df["_odds_num"] = stake * odds.fillna(0.0)
    df["_mstake_w"] = matched_stake.fillna(0.0)
    df["_modds_num"] = matched_stake.fillna(0.0) * matched_odds.fillna(0.0)

    weighted_cols = [
        col for col in ("pred", "value", "delta_time_min") if col in df.columns
    ]
    for col in weighted_cols:
        df[f"_wnum_{col}"] = stake * pd.to_numeric(df[col], errors="coerce")

    sum_cols = {"stake", "liability", "potential_profit", "matched_stake"}
    min_cols = {
        col
        for col in ("created_at", "placedDate", "marketStartDate")
        if col in df.columns
    }
    max_cols = {col for col in ("matchedDate", "settledDate") if col in df.columns}

    agg: dict[str, Any] = {}
    for col in df.columns:
        if col in group_keys or col.startswith("_"):
            continue
        if col in sum_cols:
            agg[col] = "sum"
        elif col == "profit":
            agg[col] = lambda series: series.sum(min_count=1)
        elif col in min_cols:
            agg[col] = "min"
        elif col in max_cols:
            agg[col] = "max"
        else:
            agg[col] = "first"

    agg["_stake_w"] = "sum"
    agg["_odds_num"] = "sum"
    agg["_mstake_w"] = "sum"
    agg["_modds_num"] = "sum"
    for col in weighted_cols:
        agg[f"_wnum_{col}"] = "sum"

    grouped = df.groupby(group_keys, dropna=False, sort=False).agg(agg).reset_index()

    total_stake = grouped["_stake_w"]
    stake_mask = total_stake > 0
    grouped.loc[stake_mask, "placed_odds"] = (
        grouped.loc[stake_mask, "_odds_num"] / total_stake[stake_mask]
    )

    total_mstake = grouped["_mstake_w"]
    mstake_mask = total_mstake > 0
    grouped.loc[mstake_mask, "matched_odds"] = (
        grouped.loc[mstake_mask, "_modds_num"] / total_mstake[mstake_mask]
    )

    for col in weighted_cols:
        grouped.loc[stake_mask, col] = (
            grouped.loc[stake_mask, f"_wnum_{col}"] / total_stake[stake_mask]
        )

    helper_cols = [c for c in grouped.columns if c.startswith("_")]
    grouped = grouped.drop(columns=helper_cols)

    ordered_cols = [c for c in bets.columns if c in grouped.columns]
    extra_cols = [c for c in grouped.columns if c not in ordered_cols]
    return grouped[ordered_cols + extra_cols]


@st.cache_data(ttl=120, show_spinner=False)
def load_bet_results(user_id: int | None = None) -> pd.DataFrame:
    query = f"""
        WITH bp_ranked AS (
            SELECT
                bp.ID_BET,
                bp.ID_USER,
                bp.MatchId,
                bp.ID_MARKET,
                bp.type,
                bp.bet,
                bp.bet_libelle,
                bp.odds,
                bp.stake,
                bp.status AS bet_status,
                bp.created_at,
                bp.side_back_lay,
                bp.pred,
                bp.value,
                bp.delta_time_min,
                bp.market_type AS match_type,
                bp.liability,
                bp.potential_profit,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        bp.ID_USER,
                        bp.ID_MARKET,
                        ROUND(COALESCE(bp.odds, 0), {MATCH_PRECISION}),
                        ROUND(COALESCE(bp.stake, 0), {MATCH_PRECISION})
                    ORDER BY bp.created_at, bp.ID_BET
                ) AS orbitx_match_rank
            FROM Bet_p bp
        ),
        ob_base AS (
            SELECT
                ob.ID_USER,
                ob.ID_MARKET,
                ob.ID_EVENT,
                ob.eventName,
                ob.placedDate,
                ob.matchedDate,
                ob.marketStartDate,
                ob.settledDate,
                ob.profit AS raw_profit,
                ob.size AS matched_stake,
                ob.price AS matched_odds,
                ob.offerState,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        ob.ID_USER,
                        ob.ID_MARKET,
                        ROUND(COALESCE(ob.price, 0), {MATCH_PRECISION}),
                        ROUND(COALESCE(ob.size, 0), {MATCH_PRECISION})
                    ORDER BY COALESCE(ob.matchedDate, ob.placedDate, ob.settledDate), ob.id
                ) AS orbitx_match_rank
            FROM Orbitx_bets ob
        ),
        ob_scored AS (
            SELECT
                ob_base.*,
                SUM(ob_base.raw_profit) OVER (
                    PARTITION BY ob_base.ID_USER, ob_base.ID_MARKET
                ) AS market_raw_profit,
                SUM(
                    CASE WHEN ob_base.raw_profit > 0 THEN ob_base.raw_profit ELSE 0 END
                ) OVER (
                    PARTITION BY ob_base.ID_USER, ob_base.ID_MARKET
                ) AS market_positive_profit
            FROM ob_base
        ),
        ob_ranked AS (
            SELECT
                ob_scored.ID_USER,
                ob_scored.ID_MARKET,
                ob_scored.ID_EVENT,
                ob_scored.eventName,
                ob_scored.placedDate,
                ob_scored.matchedDate,
                ob_scored.marketStartDate,
                ob_scored.settledDate,
                CASE
                    WHEN ob_scored.raw_profit > 0
                     AND ob_scored.market_raw_profit > 0
                     AND ob_scored.market_positive_profit > 0
                    THEN ob_scored.raw_profit - (
                        ob_scored.market_raw_profit * {ORBITX_COMMISSION_RATE:.2f}
                    ) * (ob_scored.raw_profit / ob_scored.market_positive_profit)
                    ELSE ob_scored.raw_profit
                END AS profit,
                ob_scored.matched_stake,
                ob_scored.matched_odds,
                ob_scored.offerState,
                ob_scored.orbitx_match_rank
            FROM ob_scored
        )
        SELECT
            bp.ID_BET,
            bp.ID_USER,
            COALESCE(u.username, CONCAT('User ', bp.ID_USER)) AS username,
            bp.MatchId,
            bp.ID_MARKET,
            bp.type,
            bp.bet,
            bp.bet_libelle,
            bp.odds AS placed_odds,
            bp.stake,
            bp.bet_status,
            bp.created_at,
            bp.side_back_lay,
            bp.pred,
            bp.value,
            bp.delta_time_min,
            bp.match_type,
            bp.liability,
            bp.potential_profit,
            ob.ID_EVENT,
            ob.eventName,
            ob.placedDate,
            ob.matchedDate,
            ob.marketStartDate,
            ob.settledDate,
            ob.profit,
            ob.matched_stake,
            ob.matched_odds,
            ob.offerState,
            aofagg.LeagueId,
            aofagg.LeagueName,
            aofagg.feed_match_date,
            aofagg.HomeTeam,
            aofagg.AwayTeam,
            aofagg.HomeTeam_clean,
            aofagg.AwayTeam_clean,
            aofagg.p_calib_home,
            aofagg.p_calib_draw,
            aofagg.p_calib_away,
            NULL AS team_name,
            NULL AS pred_odds,
            NULL AS ev_pct,
            NULL AS strategy,
            NULL AS reason,
            NULL AS analytics_matched_stake,
            NULL AS analytics_matched_odds
        FROM bp_ranked bp
        LEFT JOIN Users u
            ON bp.ID_USER = u.ID_USER
        LEFT JOIN ob_ranked ob
            ON bp.ID_USER = ob.ID_USER
           AND bp.ID_MARKET = ob.ID_MARKET
           AND ROUND(COALESCE(bp.odds, 0), {MATCH_PRECISION}) = ROUND(COALESCE(ob.matched_odds, 0), {MATCH_PRECISION})
           AND ROUND(COALESCE(bp.stake, 0), {MATCH_PRECISION}) = ROUND(COALESCE(ob.matched_stake, 0), {MATCH_PRECISION})
           AND bp.orbitx_match_rank = ob.orbitx_match_rank
        LEFT JOIN (
            SELECT
                MatchId,
                MAX(LeagueId) AS LeagueId,
                MAX(LeagueName) AS LeagueName,
                MAX(`date`) AS feed_match_date,
                MAX(HomeTeam) AS HomeTeam,
                MAX(AwayTeam) AS AwayTeam,
                MAX(HomeTeam_clean) AS HomeTeam_clean,
                MAX(AwayTeam_clean) AS AwayTeam_clean,
                MAX(p_calib_home) AS p_calib_home,
                MAX(p_calib_draw) AS p_calib_draw,
                MAX(p_calib_away) AS p_calib_away
            FROM AsianOdds_feeds
            GROUP BY MatchId
        ) aofagg
            ON CAST(aofagg.MatchId AS CHAR) = CAST(bp.MatchId AS CHAR)
        WHERE (:user_id IS NULL OR bp.ID_USER = :user_id)
        ORDER BY COALESCE(ob.settledDate, ob.matchedDate, bp.created_at) DESC, bp.ID_BET DESC
    """
    bets = _query_dataframe(query, params={"user_id": user_id})
    if bets.empty:
        return bets

    datetime_columns = [
        "created_at",
        "placedDate",
        "matchedDate",
        "marketStartDate",
        "settledDate",
        "feed_match_date",
    ]
    for column in datetime_columns:
        bets[column] = pd.to_datetime(bets[column], errors="coerce")

    numeric_columns = [
        "placed_odds",
        "stake",
        "liability",
        "potential_profit",
        "profit",
        "matched_stake",
        "matched_odds",
        "pred_odds",
        "ev_pct",
        "analytics_matched_stake",
        "analytics_matched_odds",
        "pred",
        "value",
        "delta_time_min",
        "p_calib_home",
        "p_calib_draw",
        "p_calib_away",
    ]
    for column in numeric_columns:
        bets[column] = pd.to_numeric(bets[column], errors="coerce")

    bets = _aggregate_duplicate_bets(bets)

    bets["selection"] = bets["bet_libelle"].fillna(bets["team_name"])
    bets["event_label"] = (
        bets["eventName"]
        .fillna(bets["selection"])
        .fillna(bets["MatchId"].astype("string"))
    )
    bets["pred_odds"] = bets.apply(_compute_effective_pred_odds, axis=1)
    bets["display_date"] = bets["settledDate"]
    bets["display_date"] = bets["display_date"].fillna(bets["matchedDate"])
    bets["display_date"] = bets["display_date"].fillna(bets["created_at"])
    bets["settled"] = bets["settledDate"].notna() | bets["profit"].notna()
    bets["profit"] = bets["profit"].fillna(0.0)
    bets["stake"] = bets["stake"].fillna(0.0)
    bets["potential_profit"] = bets["potential_profit"].fillna(0.0)
    bets["exposure"] = bets["liability"].fillna(bets["stake"])
    positive_stake = bets["stake"] > 0
    bets["expected_roi_pct"] = 0.0
    bets["expected_profit"] = 0.0
    bets["realized_roi_pct"] = np.nan
    bets.loc[positive_stake & bets["settled"], "realized_roi_pct"] = (
        bets.loc[positive_stake & bets["settled"], "profit"]
        / bets.loc[positive_stake & bets["settled"], "stake"]
    ) * 100
    bets["edge_pct"] = np.nan
    valid_pred = positive_stake & bets["pred_odds"].gt(0)
    bets.loc[valid_pred, "edge_pct"] = (
        (bets.loc[valid_pred, "placed_odds"] / bets.loc[valid_pred, "pred_odds"]) - 1
    ) * 100
    bets.loc[valid_pred, "expected_roi_pct"] = bets.loc[valid_pred, "edge_pct"]
    bets.loc[valid_pred, "expected_profit"] = (
        bets.loc[valid_pred, "stake"] * bets.loc[valid_pred, "expected_roi_pct"] / 100
    )

    bets["result_label"] = "Open"
    bets.loc[bets["settled"] & bets["profit"].gt(0), "result_label"] = "Win"
    bets.loc[bets["settled"] & bets["profit"].lt(0), "result_label"] = "Loss"
    bets.loc[bets["settled"] & bets["profit"].eq(0), "result_label"] = "Push"
    bets["month_label"] = bets["display_date"].dt.to_period("M").astype("string")
    bets["day_label"] = bets["display_date"].dt.date
    return bets


def compute_dashboard_kpis(bets: pd.DataFrame) -> dict[str, float]:
    if bets.empty:
        return {
            "total_bets": 0,
            "settled_bets": 0,
            "open_bets": 0,
            "total_stake": 0.0,
            "total_profit": 0.0,
            "roi_pct": 0.0,
            "open_exposure": 0.0,
            "win_rate": 0.0,
        }
    settled = bets.loc[bets["settled"]].copy()
    open_bets = bets.loc[~bets["settled"]].copy()
    total_stake = float(settled["stake"].sum())
    total_profit = float(settled["profit"].sum())
    wins = float((settled["profit"] > 0).sum())
    settled_count = int(len(settled))
    return {
        "total_bets": int(len(bets)),
        "settled_bets": settled_count,
        "open_bets": int(len(open_bets)),
        "total_stake": total_stake,
        "total_profit": total_profit,
        "roi_pct": (total_profit / total_stake * 100) if total_stake else 0.0,
        "open_exposure": float(open_bets["exposure"].sum()),
        "win_rate": (wins / settled_count * 100) if settled_count else 0.0,
    }


def build_profit_timeseries(bets: pd.DataFrame) -> pd.DataFrame:
    settled = bets.loc[bets["settled"]].copy()
    if settled.empty:
        return pd.DataFrame(
            columns=["day_label", "profit", "stake", "bets_count", "cumulative_profit"]
        )
    daily = (
        settled.groupby("day_label", as_index=False)
        .agg(
            profit=("profit", "sum"),
            stake=("stake", "sum"),
            bets_count=("ID_BET", "count"),
        )
        .sort_values("day_label")
    )
    daily["cumulative_profit"] = daily["profit"].cumsum()
    return daily


def build_match_type_summary(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame(
            columns=["match_type", "bets_count", "stake", "profit", "roi_pct"]
        )
    summary = (
        bets.groupby("match_type", dropna=False, as_index=False)
        .agg(
            bets_count=("ID_BET", "count"),
            settled_bets=("settled", "sum"),
            stake=("stake", "sum"),
            profit=("profit", "sum"),
        )
        .sort_values("profit", ascending=False)
    )
    summary["match_type"] = summary["match_type"].fillna("unknown")
    summary["roi_pct"] = 0.0
    valid_stake = summary["stake"] > 0
    summary.loc[valid_stake, "roi_pct"] = (
        summary.loc[valid_stake, "profit"] / summary.loc[valid_stake, "stake"]
    ) * 100
    return summary


def _bucket_odds(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    if value < 1.5:
        return "<1.5"
    if value < 2.0:
        return "1.5-2.0"
    if value < 2.5:
        return "2.0-2.5"
    if value < 3.0:
        return "2.5-3.0"
    if value < 5.0:
        return "3.0-5.0"
    return ">=5.0"


def _bucket_expected_roi(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    if value < 0:
        return "<0%"
    if value < 2:
        return "0-2%"
    if value < 5:
        return "2-5%"
    if value < 10:
        return "5-10%"
    if value < 20:
        return "10-20%"
    return ">=20%"


def _split_league_label(value: object) -> tuple[str, str]:
    raw = "" if value is None else " ".join(str(value).replace("*", " ").split())
    if not raw:
        return ("Unknown", "Unknown")

    def _format_label(label: str) -> str:
        uppercase_tokens = {
            "A",
            "FA",
            "HNL",
            "MLS",
            "NPL",
            "UEFA",
            "FIFA",
            "CAF",
            "AFC",
        }
        parts = []
        for token in label.split():
            if token in uppercase_tokens:
                parts.append(token)
            else:
                parts.append(token.title())
        return " ".join(parts)

    def _normalize_country(label: str) -> str:
        country_aliases = {
            "English": "England",
        }
        return country_aliases.get(label, label)

    prefix_aliases = {
        "ALBANIA": "Albania",
        "ARGENTINA": "Argentina",
        "AUSTRALIA": "Australia",
        "AUSTRIA": "Austria",
        "AZERBAIJAN": "Azerbaijan",
        "BAHRAIN": "Bahrain",
        "BELGIUM": "Belgium",
        "BOLIVIA": "Bolivia",
        "BOSNIA AND HERZEGOVINA": "Bosnia And Herzegovina",
        "BRAZIL": "Brazil",
        "BULGARIA": "Bulgaria",
        "CAF": "CAF",
        "CANADA": "Canada",
        "CHILE": "Chile",
        "CHINA": "China",
        "COLOMBIA": "Colombia",
        "CROATIA": "Croatia",
        "CZECH REPUBLIC": "Czech Republic",
        "DENMARK": "Denmark",
        "ECUADOR": "Ecuador",
        "EGYPT": "Egypt",
        "EL SALVADOR": "El Salvador",
        "ENGLISH": "England",
        "ESTONIA": "Estonia",
        "FAROE ISLANDS": "Faroe Islands",
        "FINLAND": "Finland",
        "FRANCE": "France",
        "GERMANY": "Germany",
        "GREECE": "Greece",
        "HONDURAS": "Honduras",
        "HONG KONG": "Hong Kong",
        "ICELAND": "Iceland",
        "INDIA": "India",
        "INDONESIA": "Indonesia",
        "IRELAND": "Ireland",
        "ISRAEL": "Israel",
        "ITALY": "Italy",
        "JAPAN": "Japan",
        "KAZAKHSTAN": "Kazakhstan",
        "KOREA": "Korea",
        "KUWAIT": "Kuwait",
        "LATVIA": "Latvia",
        "LITHUANIA": "Lithuania",
        "MALAYSIA": "Malaysia",
        "MEXICO": "Mexico",
        "MOROCCO": "Morocco",
        "NETHERLANDS": "Netherlands",
        "NORWAY": "Norway",
        "OMAN": "Oman",
        "PANAMA": "Panama",
        "PARAGUAY": "Paraguay",
        "PERU": "Peru",
        "POLAND": "Poland",
        "PORTUGAL": "Portugal",
        "ROMANIA": "Romania",
        "SAUDI ARABIA": "Saudi Arabia",
        "SCOTLAND": "Scotland",
        "SERBIA": "Serbia",
        "SINGAPORE": "Singapore",
        "SLOVAKIA": "Slovakia",
        "SLOVENIA": "Slovenia",
        "SOUTH AFRICA": "South Africa",
        "SPAIN": "Spain",
    }

    tokens = raw.split()
    league_starters = {
        "A",
        "ALLSVENSKAN",
        "BUNDESLIGA",
        "CHAMPIONSHIP",
        "CUP",
        "DIVISION",
        "ELITESERIEN",
        "EREDIVISIE",
        "FA",
        "FIRST",
        "HNL",
        "LA",
        "LEAGUE",
        "LIGA",
        "MLS",
        "NATIONAL",
        "PREMIER",
        "PRIMERA",
        "PRO",
        "PRVA",
        "SERIE",
        "SUPER",
        "SUPERETTAN",
    }

    split_index = None
    for index, token in enumerate(tokens):
        if token.upper() in league_starters:
            split_index = index
            break

    matched_prefix = None
    matched_length = 0
    upper_tokens = [token.upper() for token in tokens]
    for prefix, normalized in prefix_aliases.items():
        prefix_tokens = prefix.split()
        if (
            upper_tokens[: len(prefix_tokens)] == prefix_tokens
            and len(prefix_tokens) > matched_length
        ):
            matched_prefix = normalized
            matched_length = len(prefix_tokens)

    if matched_prefix is not None:
        suffix_tokens = tokens[matched_length:]
        if suffix_tokens:
            suffix = _format_label(" ".join(suffix_tokens))
            cleaned = f"{matched_prefix} {suffix}".strip()
        else:
            cleaned = matched_prefix
        return (matched_prefix, cleaned)

    if split_index is None:
        if len(tokens) == 1:
            cleaned = _normalize_country(_format_label(tokens[0]))
            return (cleaned, cleaned)
        country = _normalize_country(_format_label(tokens[0]))
        suffix = _format_label(" ".join(tokens[1:]))
        cleaned = f"{country} {suffix}".strip()
        return (country, cleaned)

    country = " ".join(tokens[:split_index]).strip()
    country_clean = _normalize_country(_format_label(country))
    suffix = _format_label(" ".join(tokens[split_index:]))
    cleaned = f"{country_clean} {suffix}".strip()
    if not country:
        country = raw
    return (country_clean, cleaned)


def _normalize_bet_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _resolve_market_side(row: pd.Series) -> str:
    selection_token = _normalize_bet_token(row.get("selection"))
    if selection_token in {"x", "draw", "matchnul", "nul", "tie", "thedraw"}:
        return "draw"

    home_token = _normalize_bet_token(row.get("HomeTeam_clean") or row.get("HomeTeam"))
    away_token = _normalize_bet_token(row.get("AwayTeam_clean") or row.get("AwayTeam"))

    if (
        selection_token
        and home_token
        and (
            selection_token == home_token
            or selection_token in home_token
            or home_token in selection_token
        )
    ):
        return "home"

    if (
        selection_token
        and away_token
        and (
            selection_token == away_token
            or selection_token in away_token
            or away_token in selection_token
        )
    ):
        return "away"

    bet_side_token = _normalize_bet_token(row.get("bet"))
    if bet_side_token in {"2", "h", "home", "1"}:
        return "home"
    if bet_side_token in {"1", "x", "d", "draw", "tie"}:
        return "draw"
    if bet_side_token in {"0", "a", "away"}:
        return "away"

    return "unknown"


def _compute_effective_pred_odds(row: pd.Series) -> float:
    side = _resolve_market_side(row)
    if side == "home":
        probability = row.get("p_calib_home")
    elif side == "draw":
        probability = row.get("p_calib_draw")
    elif side == "away":
        probability = row.get("p_calib_away")
    else:
        return np.nan

    probability = pd.to_numeric(probability, errors="coerce")
    if pd.isna(probability) or float(probability) <= 0:
        return np.nan

    prob = float(probability)
    if prob > 1.0 and prob <= 100.0:
        prob = prob / 100.0
    if prob <= 0:
        return np.nan

    return 1.0 / prob


def _clean_match_side(value: Any) -> str:
    return " ".join(str(value or "").replace("*", " ").split()).strip()


def _normalize_match_label(value: Any) -> str:
    text = _clean_match_side(value)
    if not text:
        return ""
    parts = re.split(
        r"\s+(?:v(?:s)?\.?|[-–—])\s+", text, maxsplit=1, flags=re.IGNORECASE
    )
    if len(parts) == 2:
        home = _clean_match_side(parts[0])
        away = _clean_match_side(parts[1])
        if home and away:
            return f"{home} - {away}"
    return text


def _has_match_separator(value: Any) -> bool:
    text = _clean_match_side(value)
    if not text:
        return False
    return bool(re.search(r"\s+(?:v(?:s)?\.?|[-–—])\s+", text, flags=re.IGNORECASE))


def _build_match_label(
    match_label: Any,
    home_team_clean: Any,
    away_team_clean: Any,
    home_team: Any,
    away_team: Any,
    fallback_label: Any,
) -> str:
    if _has_match_separator(match_label):
        normalized = _normalize_match_label(match_label)
        if normalized:
            return normalized

    for home_side, away_side in (
        (home_team, away_team),
        (home_team_clean, away_team_clean),
    ):
        home = _clean_match_side(home_side)
        away = _clean_match_side(away_side)
        if home and away:
            return f"{home} - {away}"

    for candidate in (match_label, fallback_label):
        normalized = _normalize_match_label(candidate)
        if normalized:
            return normalized

    return "Unknown"


def _derive_match_odds_modality(
    match_label: Any,
    selection: Any,
    bet_type: Any,
    match_type: Any,
    bet_value: Any,
) -> str:
    selection_text = str(selection or "").strip()
    if not selection_text:
        return "Unknown"

    market_label = f"{bet_type or ''} {match_type or ''}".lower()
    if "match odds" not in market_label and "1x2" not in market_label:
        return selection_text

    selection_token = _normalize_bet_token(selection_text)
    if selection_token in {"x", "draw", "matchnul", "nul", "tie"}:
        return "Draw"

    match_text = str(match_label or "").strip()
    parts = re.split(
        r"\s+(?:v(?:s)?\.?|[-–—])\s+", match_text, maxsplit=1, flags=re.IGNORECASE
    )
    if len(parts) == 2:
        home_token = _normalize_bet_token(parts[0])
        away_token = _normalize_bet_token(parts[1])
        if selection_token and (
            selection_token == home_token
            or selection_token in home_token
            or home_token in selection_token
        ):
            return "Home"
        if selection_token and (
            selection_token == away_token
            or selection_token in away_token
            or away_token in selection_token
        ):
            return "Away"

    if selection_token in {"1", "h", "home"}:
        return "Home"
    if selection_token in {"2", "a", "away"}:
        return "Away"

    bet_token = _normalize_bet_token(bet_value)
    if selection_token in {"", "1", "2", "x", "h", "a", "home", "away", "draw"}:
        if bet_token in {"1", "h", "home"}:
            return "Home"
        if bet_token in {"2", "a", "away"}:
            return "Away"
        if bet_token in {"x", "d", "draw", "tie"}:
            return "Draw"

    if "draw" in selection_text.lower():
        return "Draw"

    return selection_text


def prepare_dashboard_bets(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return bets.copy()

    dashboard = bets.copy()
    dashboard["Date"] = pd.to_datetime(dashboard["display_date"], errors="coerce")
    dashboard["Date match"] = pd.to_datetime(
        dashboard.get("feed_match_date"), errors="coerce"
    )
    dashboard["Date match"] = dashboard["Date match"].fillna(
        pd.to_datetime(dashboard["marketStartDate"], errors="coerce")
    )
    dashboard["Date match"] = dashboard["Date match"].fillna(dashboard["Date"])
    dashboard["Jour"] = dashboard["Date"].dt.strftime("%Y-%m-%d")
    dashboard["Mois"] = dashboard["Date"].dt.to_period("M").astype("string")
    week_start = dashboard["Date"].dt.to_period("W-SUN").dt.start_time
    week_end = week_start + pd.Timedelta(days=6)
    dashboard["Semaine"] = (
        week_start.dt.strftime("%d %b") + " -> " + week_end.dt.strftime("%d %b")
    )
    weekday_map = {
        0: "Lundi",
        1: "Mardi",
        2: "Mercredi",
        3: "Jeudi",
        4: "Vendredi",
        5: "Samedi",
        6: "Dimanche",
    }
    dashboard["Jour de la semaine"] = dashboard["Date"].dt.dayofweek.map(weekday_map)
    dashboard["Jour de la semaine_key"] = dashboard["Date"].dt.dayofweek
    dashboard["Mois_key"] = dashboard["Date"].dt.strftime("%Y%m")
    dashboard["Semaine_key"] = week_start.dt.strftime("%Y%m%d")

    league_parts = dashboard["LeagueName"].apply(_split_league_label)
    dashboard["Pays"] = league_parts.map(lambda values: values[0])
    dashboard["Ligue"] = league_parts.map(lambda values: values[1])

    dashboard["Match"] = dashboard.apply(
        lambda row: _build_match_label(
            row.get("event_label"),
            row.get("HomeTeam_clean"),
            row.get("AwayTeam_clean"),
            row.get("HomeTeam"),
            row.get("AwayTeam"),
            row.get("selection"),
        ),
        axis=1,
    )
    dashboard["Level"] = dashboard["strategy"].fillna("Non renseigné")
    dashboard["Round"] = dashboard["result_label"].fillna("Open")
    dashboard["Modalité"] = dashboard.apply(
        lambda row: _derive_match_odds_modality(
            row.get("Match"),
            row.get("selection"),
            row.get("type"),
            row.get("match_type"),
            row.get("bet"),
        ),
        axis=1,
    )
    dashboard["Mise"] = pd.to_numeric(dashboard["stake"], errors="coerce").fillna(0.0)
    dashboard["Cote"] = pd.to_numeric(dashboard["placed_odds"], errors="coerce")
    dashboard["Prédiction"] = pd.to_numeric(dashboard["pred_odds"], errors="coerce")
    dashboard["Gains net"] = pd.to_numeric(dashboard["profit"], errors="coerce").fillna(
        0.0
    )
    dashboard["Marge attendue"] = pd.to_numeric(
        dashboard["expected_profit"], errors="coerce"
    ).fillna(0.0)
    dashboard["EV %"] = pd.to_numeric(dashboard["ev_pct"], errors="coerce")
    dashboard["ROI attendu %"] = pd.to_numeric(
        dashboard["expected_roi_pct"], errors="coerce"
    )
    dashboard["ROI réalisé %"] = pd.to_numeric(
        dashboard["realized_roi_pct"], errors="coerce"
    )
    dashboard["Résultat"] = dashboard["result_label"].fillna("Open")
    dashboard["Etat offre"] = dashboard["offerState"].fillna("Unknown")
    dashboard["Cote_bin"] = dashboard["Cote"].map(_bucket_odds)
    dashboard["Marge_bin"] = dashboard["ROI attendu %"].map(_bucket_expected_roi)
    dashboard = dashboard.sort_values("Date", ascending=False).reset_index(drop=True)
    dashboard["Cumulative Gains"] = dashboard["Gains net"][::-1].cumsum()[::-1]
    return dashboard


def get_last_refresh_label() -> str:
    return datetime.now().strftime("%H:%M:%S")


@st.cache_data(ttl=20, show_spinner=False)
def load_upcoming_ws_odds(user_id: int | None = None) -> pd.DataFrame:
    query = """
        WITH aof_ranked AS (
            SELECT
                CAST(MatchId AS CHAR) AS match_key,
                MatchId,
                LeagueId,
                LeagueName,
                GameId,
                HomeTeam,
                HomeTeam_clean,
                AwayTeam,
                AwayTeam_clean,
                `date` AS feed_match_date,
                home_max,
                draw_max,
                away_max,
                home_pred,
                draw_pred,
                away_pred,
                p_calib_home,
                p_calib_draw,
                p_calib_away,
                ou_preds,
                hdp_preds,
                maj AS feed_maj,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(MatchId AS CHAR)
                    ORDER BY COALESCE(maj, `date`) DESC, `date` DESC
                ) AS row_rank
            FROM AsianOdds_feeds
            WHERE `date` >= NOW()
        ),
        bfl_ranked AS (
            SELECT
                ID_MARKET,
                ID_EVENT,
                MatchId AS link_match_id,
                match_title_flash,
                match_title,
                home_name AS link_home_name,
                away_name AS link_away_name,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(MatchId AS CHAR)
                    ORDER BY ID_BETFAIR DESC
                ) AS row_rank
            FROM Betfair_links_p
        ),
        ws_ranked AS (
            SELECT
                ws.*,
                ROW_NUMBER() OVER (
                    PARTITION BY ws.ID_MARKET
                    ORDER BY ws.updated_at DESC, ws.id DESC
                ) AS row_rank
            FROM WS_odds ws
        ),
        ba_market AS (
            SELECT
                ID_MARKET,
                MAX(created_at) AS analytics_at,
                AVG(pred_odds) AS analytics_pred_odds,
                MAX(ev_pct) AS analytics_ev_pct,
                MAX(strategy) AS strategy,
                MAX(reason) AS reason,
                SUM(COALESCE(matched_stake, 0)) AS analytics_matched_stake
            FROM Bet_analytics
            WHERE (:user_id IS NULL OR ID_USER = :user_id)
            GROUP BY ID_MARKET
        ),
        bp_market AS (
            SELECT
                ID_MARKET,
                COUNT(*) AS user_bets_count,
                SUM(COALESCE(stake, 0)) AS user_bets_stake,
                MAX(created_at) AS last_bet_at
            FROM Bet_p
            WHERE (:user_id IS NULL OR ID_USER = :user_id)
            GROUP BY ID_MARKET
        )
        SELECT
            ws.id,
            ws.created_at,
            ws.updated_at,
            COALESCE(ws.ID_MATCH, aof.MatchId) AS ID_MATCH,
            COALESCE(ws.ID_MARKET, bfl.ID_MARKET) AS ID_MARKET,
            bfl.ID_EVENT,
            bfl.link_match_id,
            bfl.match_title_flash,
            bfl.match_title,
            bfl.link_home_name,
            bfl.link_away_name,
            ws.home_name,
            ws.draw_name,
            ws.away_name,
            COALESCE(ws.inplay, 0) AS inplay,
            COALESCE(ws.status, 'PRED') AS status,
            CASE WHEN ws.ID_MARKET IS NULL THEN 0 ELSE 1 END AS has_ws_odds,
            ws.home_back,
            ws.home_back_1,
            ws.home_back_2,
            ws.home_lay,
            ws.home_lay_1,
            ws.home_lay_2,
            ws.draw_back,
            ws.draw_back_1,
            ws.draw_back_2,
            ws.draw_lay,
            ws.draw_lay_1,
            ws.draw_lay_2,
            ws.away_back,
            ws.away_back_1,
            ws.away_back_2,
            ws.away_lay,
            ws.away_lay_1,
            ws.away_lay_2,
            ws.home_back_size,
            ws.home_back_1_size,
            ws.home_back_2_size,
            ws.home_lay_size,
            ws.home_lay_1_size,
            ws.home_lay_2_size,
            ws.draw_back_size,
            ws.draw_back_1_size,
            ws.draw_back_2_size,
            ws.draw_lay_size,
            ws.draw_lay_1_size,
            ws.draw_lay_2_size,
            ws.away_back_size,
            ws.away_back_1_size,
            ws.away_back_2_size,
            ws.away_lay_size,
            ws.away_lay_1_size,
            ws.away_lay_2_size,
            COALESCE(ws.n_updates, 0) AS n_updates,
            aof.MatchId AS feed_match_id,
            aof.LeagueId,
            aof.LeagueName,
            aof.GameId,
            aof.HomeTeam,
            aof.HomeTeam_clean,
            aof.AwayTeam,
            aof.AwayTeam_clean,
            aof.feed_match_date,
            aof.home_max,
            aof.draw_max,
            aof.away_max,
            aof.home_pred,
            aof.draw_pred,
            aof.away_pred,
            aof.p_calib_home,
            aof.p_calib_draw,
            aof.p_calib_away,
            aof.ou_preds,
            aof.hdp_preds,
            aof.feed_maj,
            ba.analytics_at,
            ba.analytics_pred_odds,
            ba.analytics_ev_pct,
            ba.strategy,
            ba.reason,
            ba.analytics_matched_stake,
            bp.user_bets_count,
            bp.user_bets_stake,
            bp.last_bet_at
        FROM aof_ranked aof
        LEFT JOIN bfl_ranked bfl
            ON CAST(bfl.link_match_id AS CHAR) = aof.match_key
           AND bfl.row_rank = 1
        LEFT JOIN ws_ranked ws
            ON ws.ID_MARKET = bfl.ID_MARKET
           AND ws.row_rank = 1
        LEFT JOIN ba_market ba
            ON ba.ID_MARKET = bfl.ID_MARKET
        LEFT JOIN bp_market bp
            ON bp.ID_MARKET = bfl.ID_MARKET
        WHERE aof.row_rank = 1
        ORDER BY aof.feed_match_date ASC, ws.updated_at DESC
    """
    df = _query_dataframe(query, params={"user_id": user_id})
    if df.empty:
        return df

    datetime_columns = [
        "created_at",
        "updated_at",
        "feed_match_date",
        "feed_maj",
        "analytics_at",
        "last_bet_at",
    ]
    for column in datetime_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    numeric_columns = [
        "has_ws_odds",
        "inplay",
        "n_updates",
        "home_back",
        "home_back_1",
        "home_back_2",
        "home_lay",
        "home_lay_1",
        "home_lay_2",
        "draw_back",
        "draw_back_1",
        "draw_back_2",
        "draw_lay",
        "draw_lay_1",
        "draw_lay_2",
        "away_back",
        "away_back_1",
        "away_back_2",
        "away_lay",
        "away_lay_1",
        "away_lay_2",
        "home_back_size",
        "home_back_1_size",
        "home_back_2_size",
        "home_lay_size",
        "home_lay_1_size",
        "home_lay_2_size",
        "draw_back_size",
        "draw_back_1_size",
        "draw_back_2_size",
        "draw_lay_size",
        "draw_lay_1_size",
        "draw_lay_2_size",
        "away_back_size",
        "away_back_1_size",
        "away_back_2_size",
        "away_lay_size",
        "away_lay_1_size",
        "away_lay_2_size",
        "home_max",
        "draw_max",
        "away_max",
        "home_pred",
        "draw_pred",
        "away_pred",
        "p_calib_home",
        "p_calib_draw",
        "p_calib_away",
        "analytics_pred_odds",
        "analytics_ev_pct",
        "analytics_matched_stake",
        "user_bets_count",
        "user_bets_stake",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.reset_index(drop=True)


@st.cache_data(ttl=20, show_spinner=False)
def load_ws_odds_hdp() -> pd.DataFrame:
    """Latest live Asian-handicap back/lay odds per (market_id, hdp_line).

    Resolves each handicap market_id to its sibling 1X2 ``ID_MARKET`` (the key
    used by the upcoming-matches view) through ``Betfair_links_p.all_markets``.
    """
    query = """
        WITH ranked AS (
            SELECT
                h.*,
                ROW_NUMBER() OVER (
                    PARTITION BY h.market_id, h.hdp_line
                    ORDER BY h.updated_at DESC, h.id DESC
                ) AS row_rank
            FROM WS_odds_hdp h
        ),
        links AS (
            SELECT
                hdp_market_id,
                link_market_id,
                link_match_id,
                ROW_NUMBER() OVER (
                    PARTITION BY hdp_market_id
                    ORDER BY id_betfair DESC
                ) AS link_rank
            FROM (
                SELECT
                    bl.ID_BETFAIR AS id_betfair,
                    bl.ID_MARKET AS link_market_id,
                    bl.MatchId AS link_match_id,
                    jt.hdp_market_id
                FROM Betfair_links_p bl
                JOIN JSON_TABLE(
                    bl.all_markets,
                    '$[*]' COLUMNS (hdp_market_id VARCHAR(20) PATH '$.marketId')
                ) jt
            ) expanded
        )
        SELECT
            r.market_id,
            l.link_market_id,
            l.link_match_id,
            r.hdp_line,
            r.home_name,
            r.away_name,
            COALESCE(r.inplay, 0) AS inplay,
            COALESCE(r.status, 'OPEN') AS status,
            r.home_back, r.home_back_1, r.home_back_2,
            r.home_lay, r.home_lay_1, r.home_lay_2,
            r.away_back, r.away_back_1, r.away_back_2,
            r.away_lay, r.away_lay_1, r.away_lay_2,
            r.home_back_size, r.home_back_1_size, r.home_back_2_size,
            r.home_lay_size, r.home_lay_1_size, r.home_lay_2_size,
            r.away_back_size, r.away_back_1_size, r.away_back_2_size,
            r.away_lay_size, r.away_lay_1_size, r.away_lay_2_size,
            COALESCE(r.n_updates, 0) AS n_updates,
            r.updated_at
        FROM ranked r
        LEFT JOIN links l
            ON l.hdp_market_id = r.market_id
           AND l.link_rank = 1
        WHERE r.row_rank = 1
    """
    df = _query_dataframe(query)
    if df.empty:
        return df

    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    numeric_columns = [
        "hdp_line",
        "inplay",
        "n_updates",
        "home_back",
        "home_back_1",
        "home_back_2",
        "home_lay",
        "home_lay_1",
        "home_lay_2",
        "away_back",
        "away_back_1",
        "away_back_2",
        "away_lay",
        "away_lay_1",
        "away_lay_2",
        "home_back_size",
        "home_back_1_size",
        "home_back_2_size",
        "home_lay_size",
        "home_lay_1_size",
        "home_lay_2_size",
        "away_back_size",
        "away_back_1_size",
        "away_back_2_size",
        "away_lay_size",
        "away_lay_1_size",
        "away_lay_2_size",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.reset_index(drop=True)


_HDP_LINE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)(?:-(-?\d+(?:\.\d+)?))?$")


def _parse_hdp_line_label(line: object) -> tuple[float, float] | None:
    text = str(line).strip()
    match = _HDP_LINE_RE.match(text)
    if not match:
        return None
    first = float(match.group(1))
    second_raw = match.group(2)
    if second_raw is None:
        return first, first
    second = float(second_raw)
    if text.startswith("-") and second >= 0:
        second = -second
    return min(first, second), max(first, second)


def _split_handicap_line(line: float) -> tuple[float, float]:
    rounded = round(float(line) * 4.0) / 4.0
    frac = abs(rounded - int(rounded))
    if np.isclose(frac, 0.25) or np.isclose(frac, 0.75):
        return rounded - 0.25, rounded + 0.25
    return rounded, rounded


def _parse_score(result_text: object) -> tuple[float, float] | None:
    text = str(result_text or "").strip()
    match = re.search(r"(\d+)\s*:\s*(\d+)", text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _asian_leg_outcome(goal_diff: float, handicap: float, side: str) -> float:
    margin = goal_diff + float(handicap)
    if side == "away":
        margin = -margin
    if margin > 0:
        return 1.0
    if margin < 0:
        return -1.0
    return 0.0


def _asian_profit_per_unit(
    goal_diff: float,
    line: float,
    side: str,
    odds: float,
    position: str = "back",
    commission_rate: float = EXCHANGE_COMMISSION_RATE,
) -> tuple[float, float]:
    h1, h2 = _split_handicap_line(line)
    outcomes = [
        _asian_leg_outcome(goal_diff, h1, side),
        _asian_leg_outcome(goal_diff, h2, side),
    ]
    avg_outcome = float(np.mean(outcomes))

    profits = []
    for outcome in outcomes:
        if position == "lay":
            if outcome < 0:
                gross = 1.0
                profits.append(gross * (1.0 - commission_rate))
            elif outcome > 0:
                profits.append(-(float(odds) - 1.0))
            else:
                profits.append(0.0)
        else:
            if outcome > 0:
                gross = float(odds) - 1.0
                profits.append(gross * (1.0 - commission_rate))
            elif outcome < 0:
                profits.append(-1.0)
            else:
                profits.append(0.0)
    return float(np.mean(profits)), avg_outcome


def _extract_hdp_pred_pair(hdp_preds: object, line: object) -> tuple[float, float]:
    try:
        payload = (
            hdp_preds
            if isinstance(hdp_preds, dict)
            else json.loads(str(hdp_preds or "{}"))
        )
    except Exception:
        return np.nan, np.nan
    if not isinstance(payload, dict) or not payload:
        return np.nan, np.nan

    target = pd.to_numeric(line, errors="coerce")
    if pd.isna(target):
        return np.nan, np.nan
    target_value = float(target)

    best_values: dict[str, object] | None = None
    best_delta = float("inf")
    for key, values in payload.items():
        if not isinstance(values, dict):
            continue
        bounds = _parse_hdp_line_label(key)
        if bounds is None:
            continue
        midpoint = round((bounds[0] + bounds[1]) / 2.0, 3)
        delta = abs(midpoint - target_value)
        if delta < best_delta:
            best_delta = delta
            best_values = values

    if best_values is None or best_delta > 0.02:
        return np.nan, np.nan

    home_pred = pd.to_numeric(best_values.get("hdp_home_pred"), errors="coerce")
    away_pred = pd.to_numeric(best_values.get("hdp_away_pred"), errors="coerce")
    return float(home_pred) if pd.notna(home_pred) else np.nan, float(
        away_pred
    ) if pd.notna(away_pred) else np.nan


@st.cache_data(ttl=60, show_spinner=False)
def load_hdp_simulation_frame() -> pd.DataFrame:
    query = """
        WITH h_latest AS (
            SELECT
                h.*,
                ROW_NUMBER() OVER (
                    PARTITION BY h.market_id, h.hdp_line
                    ORDER BY h.updated_at DESC, h.id DESC
                ) AS row_rank
            FROM WS_odds_hdp h
            WHERE COALESCE(h.inplay, 0) = 0
        ),
        links AS (
            SELECT
                hdp_market_id,
                link_market_id,
                link_match_id,
                ROW_NUMBER() OVER (
                    PARTITION BY hdp_market_id
                    ORDER BY id_betfair DESC
                ) AS link_rank
            FROM (
                SELECT
                    bl.ID_BETFAIR AS id_betfair,
                    bl.ID_MARKET AS link_market_id,
                    bl.MatchId AS link_match_id,
                    jt.hdp_market_id
                FROM Betfair_links_p bl
                JOIN JSON_TABLE(
                    bl.all_markets,
                    '$[*]' COLUMNS (hdp_market_id VARCHAR(20) PATH '$.marketId')
                ) jt
            ) expanded
        ),
        aof_ranked AS (
            SELECT
                CAST(a.MatchId AS CHAR) AS match_key,
                a.MatchId,
                a.ID_MATCH,
                a.LeagueName,
                a.`date` AS match_date,
                a.HomeTeam_clean,
                a.AwayTeam_clean,
                a.hdp_preds,
                ROW_NUMBER() OVER (
                    PARTITION BY a.MatchId
                    ORDER BY COALESCE(a.maj, a.`date`) DESC, a.`date` DESC
                ) AS row_rank
            FROM AsianOdds_feeds a
        ),
        score_ranked AS (
            SELECT
                op.ID_MATCH,
                op.result,
                op.updated_at AS score_updated_at,
                ROW_NUMBER() OVER (
                    PARTITION BY op.ID_MATCH
                    ORDER BY op.updated_at DESC
                ) AS row_rank
            FROM Oddsportal_data op
            WHERE op.result REGEXP '^[0-9]+:[0-9]+$'
        )
        SELECT
            h.market_id,
            l.link_market_id,
            l.link_match_id,
            h.hdp_line,
            h.home_name,
            h.away_name,
            h.updated_at,
            h.status,
            h.n_updates,
            h.home_back,
            h.home_back_1,
            h.home_back_2,
            h.home_lay,
            h.home_lay_1,
            h.home_lay_2,
            h.away_back,
            h.away_back_1,
            h.away_back_2,
            h.away_lay,
            h.away_lay_1,
            h.away_lay_2,
            a.MatchId AS aof_match_id,
            a.ID_MATCH,
            a.LeagueName,
            a.match_date,
            a.HomeTeam_clean,
            a.AwayTeam_clean,
            a.hdp_preds,
            s.result,
            s.score_updated_at
        FROM h_latest h
        LEFT JOIN links l
            ON l.hdp_market_id = h.market_id
           AND l.link_rank = 1
        LEFT JOIN aof_ranked a
            ON a.match_key = CAST(l.link_match_id AS CHAR)
           AND a.row_rank = 1
        LEFT JOIN score_ranked s
            ON s.ID_MATCH = a.ID_MATCH
           AND s.row_rank = 1
        WHERE h.row_rank = 1
    """
    df = _query_dataframe(query)
    if df.empty:
        return df

    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["score_updated_at"] = pd.to_datetime(df["score_updated_at"], errors="coerce")

    numeric_columns = [
        "hdp_line",
        "n_updates",
        "home_back",
        "home_back_1",
        "home_back_2",
        "home_lay",
        "home_lay_1",
        "home_lay_2",
        "away_back",
        "away_back_1",
        "away_back_2",
        "away_lay",
        "away_lay_1",
        "away_lay_2",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["home_best_back"] = df[["home_back", "home_back_1", "home_back_2"]].max(axis=1)
    df["away_best_back"] = df[["away_back", "away_back_1", "away_back_2"]].max(axis=1)
    df["home_best_lay"] = (
        df[["home_lay", "home_lay_1", "home_lay_2"]].replace(0, np.nan).min(axis=1)
    )
    df["away_best_lay"] = (
        df[["away_lay", "away_lay_1", "away_lay_2"]].replace(0, np.nan).min(axis=1)
    )

    preds = df.apply(
        lambda row: _extract_hdp_pred_pair(row.get("hdp_preds"), row.get("hdp_line")),
        axis=1,
        result_type="expand",
    )
    preds.columns = ["home_pred_odds", "away_pred_odds"]
    df = pd.concat([df, preds], axis=1)

    df["ev_home_back_pct"] = (df["home_best_back"] / df["home_pred_odds"] - 1.0) * 100.0
    df["ev_away_back_pct"] = (df["away_best_back"] / df["away_pred_odds"] - 1.0) * 100.0
    df["ev_home_lay_pct"] = (df["home_pred_odds"] / df["home_best_lay"] - 1.0) * 100.0
    df["ev_away_lay_pct"] = (df["away_pred_odds"] / df["away_best_lay"] - 1.0) * 100.0

    scores = df["result"].apply(_parse_score)
    df["home_goals"] = scores.map(
        lambda value: value[0] if value is not None else np.nan
    )
    df["away_goals"] = scores.map(
        lambda value: value[1] if value is not None else np.nan
    )
    df["goal_diff"] = df["home_goals"] - df["away_goals"]

    def _profit_row(row: pd.Series, side: str, position: str) -> tuple[float, float]:
        goal_diff = pd.to_numeric(row.get("goal_diff"), errors="coerce")
        line = pd.to_numeric(row.get("hdp_line"), errors="coerce")
        if position == "lay":
            odds_key = "home_best_lay" if side == "home" else "away_best_lay"
        else:
            odds_key = "home_best_back" if side == "home" else "away_best_back"
        odds = pd.to_numeric(row.get(odds_key), errors="coerce")
        if pd.isna(goal_diff) or pd.isna(line) or pd.isna(odds) or float(odds) <= 1.0:
            return np.nan, np.nan
        return _asian_profit_per_unit(
            float(goal_diff),
            float(line),
            side,
            float(odds),
            position=position,
        )

    home_back_out = df.apply(
        lambda row: _profit_row(row, "home", "back"), axis=1, result_type="expand"
    )
    away_back_out = df.apply(
        lambda row: _profit_row(row, "away", "back"), axis=1, result_type="expand"
    )
    home_lay_out = df.apply(
        lambda row: _profit_row(row, "home", "lay"), axis=1, result_type="expand"
    )
    away_lay_out = df.apply(
        lambda row: _profit_row(row, "away", "lay"), axis=1, result_type="expand"
    )

    home_back_out.columns = ["home_back_profit_u", "home_back_outcome_u"]
    away_back_out.columns = ["away_back_profit_u", "away_back_outcome_u"]
    home_lay_out.columns = ["home_lay_profit_u", "home_lay_outcome_u"]
    away_lay_out.columns = ["away_lay_profit_u", "away_lay_outcome_u"]
    df = pd.concat(
        [df, home_back_out, away_back_out, home_lay_out, away_lay_out], axis=1
    )

    # Backward-compatible aliases kept for existing dashboards.
    df["home_profit_u"] = df["home_back_profit_u"]
    df["home_outcome_u"] = df["home_back_outcome_u"]
    df["away_profit_u"] = df["away_back_profit_u"]
    df["away_outcome_u"] = df["away_back_outcome_u"]

    return df.reset_index(drop=True)
