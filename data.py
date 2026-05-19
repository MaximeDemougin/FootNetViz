from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

TEAM_PLAYERS = {
    "FootNet Alpha": [
        ("Leo Garnier", "GK", 8, 40),
        ("Sacha Borel", "RB", 22, 16),
        ("Noe Rivet", "RCB", 22, 31),
        ("Ilyes Moreau", "LCB", 22, 49),
        ("Luca Perrin", "LB", 22, 64),
        ("Milo Henry", "DM", 39, 40),
        ("Enzo Rolland", "RCM", 52, 28),
        ("Nolan Picard", "LCM", 52, 52),
        ("Eliott Vasseur", "RW", 68, 16),
        ("Adam Caron", "ST", 78, 40),
        ("Theo Millet", "LW", 68, 64),
    ],
    "FootNet Horizon": [
        ("Mael Delorme", "GK", 8, 40),
        ("Rayan Faure", "RB", 22, 17),
        ("Evan Giraud", "RCB", 22, 30),
        ("Aaron Gallet", "LCB", 22, 50),
        ("Axel Dubreuil", "LB", 22, 63),
        ("Mathis Lamy", "DM", 38, 40),
        ("Yanis Brunet", "RCM", 51, 29),
        ("Nael Cousin", "LCM", 51, 51),
        ("Nino Marty", "RW", 67, 18),
        ("Lyam Arnaud", "ST", 79, 40),
        ("Timeo Mercier", "LW", 67, 62),
    ],
    "FootNet Pulse": [
        ("Tom Laval", "GK", 8, 40),
        ("Maxence Aubry", "RB", 22, 18),
        ("Soan Roussel", "RCB", 22, 30),
        ("Clement Salles", "LCB", 22, 50),
        ("Gabin Chevalier", "LB", 22, 62),
        ("Samuel Leclerc", "DM", 40, 40),
        ("Valentin Blot", "RCM", 53, 27),
        ("Antoine Coste", "LCM", 53, 53),
        ("Raphael Denis", "RW", 69, 17),
        ("Bastien David", "ST", 80, 40),
        ("Tiago Renault", "LW", 69, 63),
    ],
    "FootNet Atlas": [
        ("Arthur Gerard", "GK", 8, 40),
        ("Robin Blin", "RB", 22, 16),
        ("Jules Hardy", "RCB", 22, 31),
        ("Loan Prevost", "LCB", 22, 49),
        ("Marius Colin", "LB", 22, 64),
        ("Pablo Tessier", "DM", 40, 40),
        ("Victor Hamel", "RCM", 52, 28),
        ("Noah Bertin", "LCM", 52, 52),
        ("Tao Lejeune", "RW", 68, 16),
        ("Ethan Joly", "ST", 79, 40),
        ("Paul Brun", "LW", 68, 64),
    ],
}

ROLE_BASE = {
    "GK": {"prog": 1.5, "shot_assists": 0.1, "xt": 0.05, "touches": 32, "duels": 1.0},
    "RB": {"prog": 5.5, "shot_assists": 0.5, "xt": 0.18, "touches": 58, "duels": 4.2},
    "RCB": {"prog": 4.5, "shot_assists": 0.2, "xt": 0.12, "touches": 63, "duels": 4.8},
    "LCB": {"prog": 4.4, "shot_assists": 0.2, "xt": 0.12, "touches": 61, "duels": 4.7},
    "LB": {"prog": 5.7, "shot_assists": 0.6, "xt": 0.18, "touches": 57, "duels": 4.1},
    "DM": {"prog": 7.0, "shot_assists": 0.8, "xt": 0.26, "touches": 71, "duels": 6.8},
    "RCM": {"prog": 8.1, "shot_assists": 1.3, "xt": 0.34, "touches": 67, "duels": 6.0},
    "LCM": {"prog": 8.3, "shot_assists": 1.3, "xt": 0.35, "touches": 68, "duels": 6.1},
    "RW": {"prog": 7.5, "shot_assists": 1.8, "xt": 0.41, "touches": 52, "duels": 5.3},
    "ST": {"prog": 4.0, "shot_assists": 1.1, "xt": 0.44, "touches": 43, "duels": 6.4},
    "LW": {"prog": 7.6, "shot_assists": 1.9, "xt": 0.42, "touches": 51, "duels": 5.2},
}

OPPONENTS = [
    "Olympique Nord",
    "Racing Sud",
    "Union Est",
    "Stade Ocean",
    "FC Belvedere",
    "AS Forge",
    "SC Lumiere",
    "US Voltige",
]

ROLE_LAYOUT = {
    "GK": (8, 40),
    "RB": (22, 16),
    "RCB": (22, 31),
    "CB": (22, 40),
    "LCB": (22, 49),
    "LB": (22, 64),
    "DM": (39, 40),
    "CM": (52, 40),
    "RCM": (52, 28),
    "LCM": (52, 52),
    "AM": (63, 40),
    "RM": (63, 22),
    "LM": (63, 58),
    "RW": (68, 16),
    "ST": (78, 40),
    "FW": (78, 40),
    "LW": (68, 64),
}

DEFAULT_REAL_DATA_DIR = Path(__file__).resolve().parent / "data" / "real"
REAL_DATA_DIR_ENV_VAR = "FOOTNET_DATA_DIR"
SUPPORTED_DATA_EXTENSIONS = (".parquet", ".csv", ".json")

DATASET_ALIASES = {
    "matches": {
        "team": ["team_name", "club", "squad"],
        "match_id": ["fixture_id", "game_id", "id_match"],
        "match_day": ["matchday", "gameweek", "round", "journee"],
        "opponent": ["opponent_name", "adversaire", "opposition"],
        "score_for": ["goals_for", "team_goals"],
        "score_against": ["goals_against", "opponent_goals"],
        "xg_for": ["xg", "team_xg", "xgf"],
        "xg_against": ["opponent_xg", "xga", "xg_allowed"],
        "possession": ["possession_pct", "ball_possession"],
        "pass_accuracy": ["pass_accuracy_pct", "pass_pct", "passing_accuracy"],
        "press_intensity": ["ppda", "pressing_intensity"],
        "final_third_entries": ["final_3rd_entries", "entries_final_third", "box_entries"],
        "scoreline": ["score", "result", "score_line"],
    },
    "passes": {
        "team": ["team_name", "club", "squad"],
        "match_id": ["fixture_id", "game_id", "id_match"],
        "match_day": ["matchday", "gameweek", "round", "journee"],
        "passer": ["from", "source", "player_from", "passer_name"],
        "receiver": ["to", "target", "player_to", "receiver_name"],
        "count": ["pass_count", "passes", "volume", "weight"],
        "passer_role": ["from_role", "source_role", "passer_position"],
        "receiver_role": ["to_role", "target_role", "receiver_position"],
        "passer_x": ["from_x", "source_x", "start_x"],
        "passer_y": ["from_y", "source_y", "start_y"],
        "receiver_x": ["to_x", "target_x", "end_x"],
        "receiver_y": ["to_y", "target_y", "end_y"],
    },
    "players": {
        "team": ["team_name", "club", "squad"],
        "match_id": ["fixture_id", "game_id", "id_match"],
        "match_day": ["matchday", "gameweek", "round", "journee"],
        "player": ["player_name", "name"],
        "role": ["position", "poste"],
        "minutes": ["mins", "minutes_played"],
        "touches": ["touch_count", "ball_touches"],
        "progressive_passes": ["prog_passes", "progressive_pass_count"],
        "shot_assists": ["key_passes", "passes_to_shot"],
        "xT_added": ["xt", "xt_added", "threat_added"],
        "duels_won": ["duels", "duels_success"],
        "x": ["avg_x", "position_x"],
        "y": ["avg_y", "position_y"],
    },
    "roster": {
        "team": ["team_name", "club", "squad"],
        "player": ["player_name", "name"],
        "role": ["position", "poste"],
        "x": ["avg_x", "position_x"],
        "y": ["avg_y", "position_y"],
    },
}

DATASET_DEFAULTS = {
    "matches": {
        "match_id": pd.NA,
        "xg_for": 0.0,
        "xg_against": 0.0,
        "possession": 0.0,
        "pass_accuracy": 0.0,
        "press_intensity": 0.0,
        "final_third_entries": 0,
        "score_for": pd.NA,
        "score_against": pd.NA,
        "scoreline": "-",
    },
    "passes": {
        "match_id": pd.NA,
        "count": 0,
        "passer_role": pd.NA,
        "receiver_role": pd.NA,
        "passer_x": pd.NA,
        "passer_y": pd.NA,
        "receiver_x": pd.NA,
        "receiver_y": pd.NA,
    },
    "players": {
        "match_id": pd.NA,
        "role": pd.NA,
        "minutes": 0,
        "touches": 0,
        "progressive_passes": 0,
        "shot_assists": 0.0,
        "xT_added": 0.0,
        "duels_won": 0.0,
        "x": pd.NA,
        "y": pd.NA,
    },
    "roster": {
        "role": pd.NA,
        "x": pd.NA,
        "y": pd.NA,
    },
}

DATASET_REQUIRED_COLUMNS = {
    "matches": ["team", "opponent"],
    "passes": ["team", "passer", "receiver", "count"],
    "players": ["team", "player"],
    "roster": ["team", "player"],
}

DATASET_INT_COLUMNS = {
    "matches": ["match_day", "final_third_entries"],
    "passes": ["match_day", "count"],
    "players": ["match_day", "minutes", "touches", "progressive_passes"],
    "roster": [],
}

DATASET_FLOAT_COLUMNS = {
    "matches": ["xg_for", "xg_against", "possession", "pass_accuracy", "press_intensity"],
    "passes": ["passer_x", "passer_y", "receiver_x", "receiver_y"],
    "players": ["shot_assists", "xT_added", "duels_won", "x", "y"],
    "roster": ["x", "y"],
}

KEY_TEXT_COLUMNS = {
    "matches": ["team", "match_id", "opponent", "scoreline"],
    "passes": ["team", "match_id", "passer", "receiver", "passer_role", "receiver_role"],
    "players": ["team", "match_id", "player", "role"],
    "roster": ["team", "player", "role"],
}


@st.cache_data(show_spinner=False)
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260519)
    match_rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    roster_rows: list[dict[str, Any]] = []

    for team_index, (team, players) in enumerate(TEAM_PLAYERS.items()):
        for player_name, role, x_pos, y_pos in players:
            roster_rows.append(
                {
                    "team": team,
                    "player": player_name,
                    "role": role,
                    "x": x_pos,
                    "y": y_pos,
                }
            )

        base_possession = 53 + team_index * 1.8
        base_accuracy = 84.5 + team_index * 1.1
        base_xg = 1.55 + team_index * 0.15

        for match_day in range(1, 9):
            opponent = OPPONENTS[(match_day + team_index) % len(OPPONENTS)]
            possession = float(np.clip(rng.normal(base_possession, 2.8), 46, 66))
            pass_accuracy = float(np.clip(rng.normal(base_accuracy, 1.8), 79, 93))
            xg_for = float(np.clip(rng.normal(base_xg, 0.24), 0.9, 2.6))
            xg_against = float(np.clip(rng.normal(1.12 - team_index * 0.04, 0.18), 0.45, 1.85))
            press_intensity = float(np.clip(rng.normal(8.2 + team_index * 0.35, 0.9), 5.5, 12.0))
            final_third_entries = int(np.clip(rng.normal(29 + team_index * 1.4, 4.5), 18, 44))
            score_for = max(0, int(round(xg_for + rng.normal(0.0, 0.55))))
            score_against = max(0, int(round(xg_against + rng.normal(0.0, 0.45))))

            match_rows.append(
                {
                    "team": team,
                    "match_day": match_day,
                    "opponent": opponent,
                    "xg_for": round(xg_for, 2),
                    "xg_against": round(xg_against, 2),
                    "possession": round(possession, 1),
                    "pass_accuracy": round(pass_accuracy, 1),
                    "press_intensity": round(press_intensity, 1),
                    "final_third_entries": final_third_entries,
                    "scoreline": f"{score_for}-{score_against}",
                }
            )

            for player_name, role, x_pos, y_pos in players:
                role_base = ROLE_BASE[role]
                player_rows.append(
                    {
                        "team": team,
                        "match_day": match_day,
                        "player": player_name,
                        "role": role,
                        "minutes": int(np.clip(rng.normal(84, 6), 62, 90)),
                        "touches": int(np.clip(rng.normal(role_base["touches"], 7), 18, 98)),
                        "progressive_passes": int(np.clip(rng.normal(role_base["prog"], 1.6), 0, 15)),
                        "shot_assists": float(np.clip(rng.normal(role_base["shot_assists"], 0.45), 0, 3.4)),
                        "xT_added": float(np.clip(rng.normal(role_base["xt"], 0.09), 0.01, 0.7)),
                        "duels_won": float(np.clip(rng.normal(role_base["duels"], 1.25), 0, 12)),
                        "x": x_pos,
                        "y": y_pos,
                    }
                )

            player_names = [player_name for player_name, _, _, _ in players]
            player_lookup = {player_name: (role, x_pos, y_pos) for player_name, role, x_pos, y_pos in players}

            for passer_name, passer_role, passer_x, passer_y in players:
                receiver_count = 3 if passer_role == "GK" else 4
                receiver_names = rng.choice(
                    [name for name in player_names if name != passer_name],
                    size=receiver_count,
                    replace=False,
                )
                role_bias = 1.4 if passer_role in {"DM", "RCM", "LCM"} else 0.0
                for receiver_name in receiver_names:
                    receiver_role, receiver_x, receiver_y = player_lookup[receiver_name]
                    connection_bias = 1.8 if receiver_role in {"RW", "LW", "ST"} else 0.0
                    pass_count = int(
                        np.clip(
                            rng.normal(6.5 + role_bias + connection_bias, 2.0),
                            1,
                            18,
                        )
                    )
                    pass_rows.append(
                        {
                            "team": team,
                            "match_day": match_day,
                            "passer": passer_name,
                            "receiver": receiver_name,
                            "count": pass_count,
                            "passer_role": passer_role,
                            "receiver_role": receiver_role,
                            "passer_x": passer_x,
                            "passer_y": passer_y,
                            "receiver_x": receiver_x,
                            "receiver_y": receiver_y,
                        }
                    )

    return (
        pd.DataFrame(match_rows),
        pd.DataFrame(pass_rows),
        pd.DataFrame(player_rows),
        pd.DataFrame(roster_rows),
    )


def _real_data_dir() -> Path:
    configured_path = os.getenv(REAL_DATA_DIR_ENV_VAR, "").strip()
    return Path(configured_path).expanduser() if configured_path else DEFAULT_REAL_DATA_DIR


def _find_dataset_file(dataset_name: str, base_dir: Path) -> Path | None:
    for extension in SUPPORTED_DATA_EXTENSIONS:
        candidate = base_dir / f"{dataset_name}{extension}"
        if candidate.exists():
            return candidate
    return None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    raise ValueError(f"Format non supporte pour {path.name}")


def _rename_aliases(frame: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    renamed = frame.copy()
    lower_to_actual = {str(column).lower(): column for column in renamed.columns}
    replacements: dict[str, str] = {}
    for canonical_name, alternatives in aliases.items():
        if canonical_name in renamed.columns:
            continue
        for alias in [canonical_name, *alternatives]:
            actual = lower_to_actual.get(alias.lower())
            if actual is not None:
                replacements[actual] = canonical_name
                break
    return renamed.rename(columns=replacements)


def _clean_text_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in columns:
        if column not in cleaned.columns:
            continue
        string_values = cleaned[column].astype("string").str.strip()
        cleaned[column] = string_values.mask(string_values.isin(["", "nan", "None", "<NA>"]))
    return cleaned


def _coerce_numeric_columns(frame: pd.DataFrame, int_columns: list[str], float_columns: list[str]) -> pd.DataFrame:
    coerced = frame.copy()
    for column in float_columns:
        if column in coerced.columns:
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    for column in int_columns:
        if column in coerced.columns:
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce").fillna(0).round().astype(int)
    return coerced


def _normalize_dataset(dataset_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _rename_aliases(frame, DATASET_ALIASES[dataset_name])
    normalized = _clean_text_columns(normalized, KEY_TEXT_COLUMNS[dataset_name])

    for column, default_value in DATASET_DEFAULTS[dataset_name].items():
        if column not in normalized.columns:
            normalized[column] = default_value

    missing_columns = [
        column for column in DATASET_REQUIRED_COLUMNS[dataset_name] if column not in normalized.columns
    ]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans {dataset_name}: {', '.join(missing_columns)}")

    if dataset_name in {"matches", "passes", "players"}:
        has_match_day = "match_day" in normalized.columns
        has_match_id = "match_id" in normalized.columns
        if not has_match_day and not has_match_id:
            raise ValueError(f"{dataset_name} doit contenir match_day ou match_id")

    normalized = _coerce_numeric_columns(
        normalized,
        DATASET_INT_COLUMNS[dataset_name],
        DATASET_FLOAT_COLUMNS[dataset_name],
    )

    if dataset_name == "matches" and {"score_for", "score_against"}.issubset(normalized.columns):
        score_for = pd.to_numeric(normalized["score_for"], errors="coerce")
        score_against = pd.to_numeric(normalized["score_against"], errors="coerce")
        scoreline = score_for.fillna(-1).astype(int).astype(str) + "-" + score_against.fillna(-1).astype(int).astype(str)
        normalized["scoreline"] = normalized["scoreline"].where(normalized["scoreline"].ne("-"), scoreline)

    return normalized


def _resolve_match_days(
    matches_df: pd.DataFrame,
    passes_df: pd.DataFrame,
    players_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = matches_df.copy()
    passes = passes_df.copy()
    players = players_df.copy()

    if "match_day" not in matches.columns or matches["match_day"].isna().all():
        sort_columns = ["team"]
        if "date" in matches.columns:
            sort_columns.append("date")
        elif "match_id" in matches.columns:
            sort_columns.append("match_id")
        else:
            sort_columns.append("opponent")
        matches = matches.sort_values(sort_columns).reset_index(drop=True)
        matches["match_day"] = matches.groupby("team").cumcount() + 1

    if matches["match_day"].isna().any():
        missing_mask = matches["match_day"].isna()
        filled_days = matches.loc[missing_mask].groupby("team").cumcount() + 1
        matches.loc[missing_mask, "match_day"] = filled_days.to_numpy()

    matches["match_day"] = pd.to_numeric(matches["match_day"], errors="coerce").fillna(0).round().astype(int)
    match_lookup = matches[["team", "match_day"]].copy()
    if "match_id" in matches.columns:
        match_lookup["match_id"] = matches["match_id"]

    def attach_match_day(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        attached = frame.copy()
        if "match_day" in attached.columns and attached["match_day"].notna().any():
            attached["match_day"] = pd.to_numeric(attached["match_day"], errors="coerce")
            missing_match_day = attached["match_day"].isna()
        else:
            attached["match_day"] = pd.NA
            missing_match_day = pd.Series(True, index=attached.index)

        if missing_match_day.any() and "match_id" in attached.columns and "match_id" in match_lookup.columns:
            merged = attached.merge(
                match_lookup[["team", "match_id", "match_day"]].drop_duplicates(),
                on=["team", "match_id"],
                how="left",
                suffixes=("", "_lookup"),
            )
            attached = merged
            resolved_match_day = pd.to_numeric(attached["match_day"], errors="coerce")
            lookup_match_day = pd.to_numeric(attached["match_day_lookup"], errors="coerce")
            attached["match_day"] = resolved_match_day.where(resolved_match_day.notna(), lookup_match_day)
            attached = attached.drop(columns=["match_day_lookup"])

        if attached["match_day"].isna().any():
            raise ValueError(f"Impossible de resoudre match_day pour {label}")

        attached["match_day"] = pd.to_numeric(attached["match_day"], errors="coerce").fillna(0).round().astype(int)
        return attached

    passes = attach_match_day(passes, "passes")
    players = attach_match_day(players, "players")
    return matches, passes, players


def _role_position(role: Any) -> tuple[float, float]:
    role_key = str(role or "").upper()
    return ROLE_LAYOUT.get(role_key, (50.0, 40.0))


def _build_roster_from_players(players_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        players_df.groupby(["team", "player"], as_index=False)
        .agg({"role": "first", "x": "median", "y": "median"})
        .reset_index(drop=True)
    )
    return grouped


def _fill_positions_from_role(frame: pd.DataFrame, role_column: str, x_column: str, y_column: str) -> pd.DataFrame:
    filled = frame.copy()
    fallback_positions = filled[role_column].map(_role_position)
    fallback_x = fallback_positions.map(lambda item: item[0])
    fallback_y = fallback_positions.map(lambda item: item[1])
    filled[x_column] = pd.to_numeric(filled[x_column], errors="coerce").fillna(fallback_x)
    filled[y_column] = pd.to_numeric(filled[y_column], errors="coerce").fillna(fallback_y)
    return filled


def _prepare_roster(roster_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    roster = roster_df.copy() if not roster_df.empty else _build_roster_from_players(players_df)
    if roster.empty:
        raise ValueError("Le roster FootNet est vide et ne peut pas etre derive depuis players.")
    roster["role"] = roster["role"].fillna("UNK")
    roster = _fill_positions_from_role(roster, "role", "x", "y")
    roster = roster.drop_duplicates(subset=["team", "player"], keep="first").reset_index(drop=True)
    return roster


def _enrich_players(players_df: pd.DataFrame, roster_df: pd.DataFrame) -> pd.DataFrame:
    roster_lookup = roster_df.rename(columns={"role": "roster_role", "x": "roster_x", "y": "roster_y"})
    enriched = players_df.merge(roster_lookup, on=["team", "player"], how="left")
    enriched["role"] = enriched["role"].fillna(enriched["roster_role"]).fillna("UNK")
    enriched["x"] = pd.to_numeric(enriched["x"], errors="coerce").fillna(enriched["roster_x"])
    enriched["y"] = pd.to_numeric(enriched["y"], errors="coerce").fillna(enriched["roster_y"])
    enriched = _fill_positions_from_role(enriched, "role", "x", "y")
    return enriched.drop(columns=["roster_role", "roster_x", "roster_y"])


def _enrich_passes(passes_df: pd.DataFrame, roster_df: pd.DataFrame) -> pd.DataFrame:
    passer_lookup = roster_df.rename(
        columns={
            "player": "passer",
            "role": "passer_role_lookup",
            "x": "passer_x_lookup",
            "y": "passer_y_lookup",
        }
    )
    receiver_lookup = roster_df.rename(
        columns={
            "player": "receiver",
            "role": "receiver_role_lookup",
            "x": "receiver_x_lookup",
            "y": "receiver_y_lookup",
        }
    )

    enriched = passes_df.merge(
        passer_lookup[["team", "passer", "passer_role_lookup", "passer_x_lookup", "passer_y_lookup"]],
        on=["team", "passer"],
        how="left",
    )
    enriched = enriched.merge(
        receiver_lookup[
            ["team", "receiver", "receiver_role_lookup", "receiver_x_lookup", "receiver_y_lookup"]
        ],
        on=["team", "receiver"],
        how="left",
    )

    enriched["passer_role"] = enriched["passer_role"].fillna(enriched["passer_role_lookup"]).fillna("UNK")
    enriched["receiver_role"] = (
        enriched["receiver_role"].fillna(enriched["receiver_role_lookup"]).fillna("UNK")
    )
    enriched["passer_x"] = pd.to_numeric(enriched["passer_x"], errors="coerce").fillna(enriched["passer_x_lookup"])
    enriched["passer_y"] = pd.to_numeric(enriched["passer_y"], errors="coerce").fillna(enriched["passer_y_lookup"])
    enriched["receiver_x"] = pd.to_numeric(enriched["receiver_x"], errors="coerce").fillna(
        enriched["receiver_x_lookup"]
    )
    enriched["receiver_y"] = pd.to_numeric(enriched["receiver_y"], errors="coerce").fillna(
        enriched["receiver_y_lookup"]
    )
    enriched = _fill_positions_from_role(enriched, "passer_role", "passer_x", "passer_y")
    enriched = _fill_positions_from_role(enriched, "receiver_role", "receiver_x", "receiver_y")
    return enriched.drop(
        columns=[
            "passer_role_lookup",
            "passer_x_lookup",
            "passer_y_lookup",
            "receiver_role_lookup",
            "receiver_x_lookup",
            "receiver_y_lookup",
        ]
    )


def _load_real_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_dir = _real_data_dir()
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Aucun dossier de donnees reelles trouve dans {base_dir}."
        )

    missing_datasets = [
        dataset_name for dataset_name in ["matches", "passes", "players"] if _find_dataset_file(dataset_name, base_dir) is None
    ]
    if missing_datasets:
        raise FileNotFoundError(
            "Fichiers FootNet manquants: " + ", ".join(f"{name}.csv|json|parquet" for name in missing_datasets)
        )

    matches_df = _normalize_dataset("matches", _read_table(_find_dataset_file("matches", base_dir)))
    passes_df = _normalize_dataset("passes", _read_table(_find_dataset_file("passes", base_dir)))
    players_df = _normalize_dataset("players", _read_table(_find_dataset_file("players", base_dir)))
    matches_df, passes_df, players_df = _resolve_match_days(matches_df, passes_df, players_df)

    roster_file = _find_dataset_file("roster", base_dir)
    roster_df = _normalize_dataset("roster", _read_table(roster_file)) if roster_file else pd.DataFrame()

    roster_df = _prepare_roster(roster_df, players_df)
    players_df = _enrich_players(players_df, roster_df)
    passes_df = _enrich_passes(passes_df, roster_df)
    matches_df = matches_df.sort_values(["team", "match_day"]).reset_index(drop=True)
    passes_df = passes_df.sort_values(["team", "match_day", "passer", "receiver"]).reset_index(drop=True)
    players_df = players_df.sort_values(["team", "match_day", "player"]).reset_index(drop=True)
    return matches_df, passes_df, players_df, roster_df


@st.cache_data(show_spinner=False)
def load_data_bundle() -> dict[str, Any]:
    data_dir = _real_data_dir()
    try:
        matches_df, passes_df, players_df, roster_df = _load_real_data()
        return {
            "source": "real",
            "path": str(data_dir),
            "reason": "",
            "matches": matches_df,
            "passes": passes_df,
            "players": players_df,
            "roster": roster_df,
        }
    except Exception as exc:
        matches_df, passes_df, players_df, roster_df = load_demo_data()
        return {
            "source": "demo",
            "path": str(data_dir),
            "reason": str(exc),
            "matches": matches_df,
            "passes": passes_df,
            "players": players_df,
            "roster": roster_df,
        }


def _loaded_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bundle = load_data_bundle()
    return bundle["matches"], bundle["passes"], bundle["players"], bundle["roster"]


def get_data_source_status() -> dict[str, str]:
    bundle = load_data_bundle()
    matches_df = bundle["matches"]
    teams_count = str(matches_df["team"].nunique()) if not matches_df.empty else "0"
    matches_count = str(len(matches_df))
    return {
        "source": bundle["source"],
        "path": bundle["path"],
        "reason": bundle["reason"],
        "teams_count": teams_count,
        "matches_count": matches_count,
    }


def list_teams() -> list[str]:
    matches_df, _, _, _ = _loaded_frames()
    teams = sorted(matches_df["team"].dropna().unique().tolist())
    return teams if teams else list(TEAM_PLAYERS.keys())


def get_team_matches(team: str) -> pd.DataFrame:
    matches_df, _, _, _ = _loaded_frames()
    return matches_df.loc[matches_df["team"] == team].sort_values("match_day").reset_index(drop=True)


def _trend_delta(frame: pd.DataFrame, column: str) -> float:
    if len(frame) < 4:
        return 0.0
    head_mean = frame.head(3)[column].mean()
    tail_mean = frame.tail(3)[column].mean()
    return round(tail_mean - head_mean, 2)


def get_team_kpis(team: str) -> dict[str, float]:
    team_matches = get_team_matches(team)
    return {
        "xg_for": round(team_matches["xg_for"].mean(), 2),
        "xg_for_delta": _trend_delta(team_matches, "xg_for"),
        "possession": round(team_matches["possession"].mean(), 1),
        "possession_delta": _trend_delta(team_matches, "possession"),
        "pass_accuracy": round(team_matches["pass_accuracy"].mean(), 1),
        "pass_accuracy_delta": _trend_delta(team_matches, "pass_accuracy"),
        "press_intensity": round(team_matches["press_intensity"].mean(), 1),
        "press_intensity_delta": _trend_delta(team_matches, "press_intensity"),
    }


def get_comparison_frame(primary_team: str, comparison_team: str) -> pd.DataFrame:
    matches_df, _, _, _ = _loaded_frames()
    metrics = (
        matches_df.groupby("team", as_index=False)[
            ["xg_for", "possession", "pass_accuracy", "press_intensity", "final_third_entries"]
        ]
        .mean()
        .round(2)
    )
    return metrics.loc[metrics["team"].isin([primary_team, comparison_team])].reset_index(drop=True)


def get_top_combinations(team: str, match_day: int | None = None, top_n: int = 8) -> pd.DataFrame:
    _, passes_df, _, _ = _loaded_frames()
    frame = passes_df.loc[passes_df["team"] == team]
    if match_day is not None:
        frame = frame.loc[frame["match_day"] == match_day]
    top_links = (
        frame.groupby(["passer", "receiver"], as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    top_links["connection"] = top_links["passer"] + " -> " + top_links["receiver"]
    return top_links[["connection", "count", "passer", "receiver"]]


def get_team_network(
    team: str,
    match_day: int | None = None,
    min_connections: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, passes_df, players_df, roster_df = _loaded_frames()
    team_passes = passes_df.loc[passes_df["team"] == team]
    team_players = players_df.loc[players_df["team"] == team]

    if match_day is not None:
        team_passes = team_passes.loc[team_passes["match_day"] == match_day]
        team_players = team_players.loc[team_players["match_day"] == match_day]

    edges = (
        team_passes.groupby(
            [
                "passer",
                "receiver",
                "passer_x",
                "passer_y",
                "receiver_x",
                "receiver_y",
            ],
            as_index=False,
        )["count"]
        .sum()
        .sort_values("count", ascending=False)
    )
    edges = edges.loc[edges["count"] >= min_connections].reset_index(drop=True)

    outgoing = team_passes.groupby("passer")["count"].sum().rename("outgoing")
    incoming = team_passes.groupby("receiver")["count"].sum().rename("incoming")
    touches = outgoing.add(incoming, fill_value=0).rename("touches")
    player_load = (
        team_players.groupby(["player", "role"], as_index=False)[
            ["touches", "progressive_passes", "xT_added"]
        ]
        .mean()
        .round(2)
    )
    roster = roster_df.loc[roster_df["team"] == team].copy()
    nodes = (
        roster.merge(player_load, on="player", how="left")
        .merge(touches.rename("network_touches"), left_on="player", right_index=True, how="left")
        .fillna({"touches": 0, "progressive_passes": 0, "xT_added": 0, "network_touches": 0})
    )
    nodes["node_size"] = 16 + nodes["network_touches"] * 0.55
    return nodes, edges


def get_player_metrics(team: str) -> pd.DataFrame:
    _, passes_df, players_df, _ = _loaded_frames()
    team_players = players_df.loc[players_df["team"] == team]
    outgoing = passes_df.loc[passes_df["team"] == team].groupby("passer")["count"].sum().rename("passes_played")
    incoming = passes_df.loc[passes_df["team"] == team].groupby("receiver")["count"].sum().rename("passes_received")

    summary = (
        team_players.groupby(["player", "role"], as_index=False)[
            ["minutes", "touches", "progressive_passes", "shot_assists", "xT_added", "duels_won"]
        ]
        .sum()
        .round(2)
    )
    summary = summary.merge(outgoing, left_on="player", right_index=True, how="left")
    summary = summary.merge(incoming, left_on="player", right_index=True, how="left")
    summary = summary.fillna({"passes_played": 0, "passes_received": 0})
    summary["influence_index"] = (
        summary["xT_added"] * 14
        + summary["progressive_passes"] * 0.8
        + summary["shot_assists"] * 4
        + summary["duels_won"] * 0.6
    ).round(1)
    return summary.sort_values("influence_index", ascending=False).reset_index(drop=True)


def get_available_match_labels(team: str) -> dict[str, int | None]:
    matches = get_team_matches(team)
    labels: dict[str, int | None] = {"Tous les matchs": None}
    for row in matches.itertuples(index=False):
        labels[f"J{row.match_day} vs {row.opponent} ({row.scoreline})"] = int(row.match_day)
    return labels