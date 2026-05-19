import pandas as pd
import streamlit as st

from betting_data import get_db_status, get_user_catalog
from dashboard_view import render_dashboard
from upcoming_matches_view import render_upcoming_matches
from ui import apply_theme

st.set_page_config(page_title="FootNetViz", page_icon="⚽", layout="wide")
apply_theme()

db_status = get_db_status()
users = get_user_catalog() if db_status["connected"] == "true" else None

if users is not None and not users.empty:
    valid_user_ids = users["ID_USER"].astype(int).tolist()
    if st.session_state.get("selected_user_id") not in valid_user_ids:
        st.session_state.selected_user_id = valid_user_ids[0]
else:
    st.session_state.selected_user_id = None

with st.sidebar:
    st.markdown("## FootNet")

    selected_page = st.radio(
        "Page",
        ["Dashboard", "Matchs a venir"],
        key="selected_page",
    )

    if db_status["connected"] == "true":
        if users is not None and not users.empty:
            options = users["ID_USER"].astype(int).tolist()
            labels = dict(zip(options, users["label"].tolist(), strict=False))
            st.selectbox(
                "Utilisateur",
                options,
                key="selected_user_id",
                format_func=lambda user_id: labels.get(user_id, f"User {user_id}"),
            )
            selected_row = users.loc[
                users["ID_USER"].astype(int) == int(st.session_state.selected_user_id)
            ].iloc[0]
            st.caption(
                f"Derniere activite: {selected_row['last_activity']:%Y-%m-%d %H:%M}"
                if pd.notna(selected_row["last_activity"])
                else "Derniere activite indisponible"
            )
        else:
            st.warning("Aucun utilisateur trouve dans Bet_p.")
    else:
        st.error("Connexion BDD indisponible")
        st.caption(db_status["reason"])
        st.caption(
            "Ajoutez `db_url` dans `.streamlit/secrets.toml` ou exportez `FOOTNET_DB_URL`."
        )

if st.session_state.get("selected_page") == "Matchs a venir":
    render_upcoming_matches()
else:
    render_dashboard()
