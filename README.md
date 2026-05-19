# FootNetViz

FootNetViz est maintenant un dashboard Streamlit branche en live sur la base FootNet pour suivre les resultats `Bet_p`, dans un esprit proche de TeNNetViz: preparation SQL, enrichissement metier, puis lecture cockpit des performances.

## Ce que contient cette premiere version

- un dashboard unique des paris `Bet_p`
- une connexion MySQL via secret Streamlit `db_url`
- une jointure entre `Bet_p`, `Orbitx_bets`, `Bet_analytics` et `Users`
- des KPI de PnL, ROI, exposition et win rate
- des graphes de PnL cumule, PnL par `match_type` et detail des paris

## Lancer le projet

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .streamlit
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
# puis renseigner db_url
streamlit run app.py
```

## Configuration du secret

Le dashboard lit en priorite:

- `db_url` depuis `.streamlit/secrets.toml`
- sinon `FOOTNET_DB_URL` depuis l'environnement

Le nom de base utilise est `FootNet` par defaut. Vous pouvez aussi definir `db_name` ou `FOOTNET_DB_NAME` si besoin.

Exemple minimal:

```toml
db_url = "mysql+pymysql://user:password@host:port/"
db_name = "FootNet"
```

## Modele de donnees exploite

La vue actuelle part de:

- `Bet_p` pour les ordres / stakes / odds / potentiel
- `Orbitx_bets` pour le profit realise
- `Bet_analytics` pour `pred_odds`, `ev_pct`, `strategy`, `reason`
- `Users` pour l'identite utilisateur

Jointure principale retenue:

```sql
Bet_p.ID_USER = Orbitx_bets.ID_USER
AND Bet_p.ID_MARKET = Orbitx_bets.ID_MARKET
```

Cette jointure est la plus fiable pour afficher les resultats reels du `bet_p`.

## Exploration deja faite

Les tables principales detectees dans FootNet cote betting sont:

- `Bet_p`
- `Orbitx_bets`
- `Bet_analytics`
- `Betfair_links_p`
- `Users`

Le dashboard live charge actuellement 576 lignes `Bet_p` pour l'utilisateur detecte en test et les enrichit avec les profits realises Orbitx.

## Stack

- Python 3.12+
- Streamlit
- Pandas
- Plotly
- SQLAlchemy
- PyMySQL

Cette version ne copie pas le code de TeNNetViz; elle reprend surtout son schema de travail: preparation des donnees de paris dans la couche Python, puis restitution dans un dashboard analytique simple a faire evoluer.