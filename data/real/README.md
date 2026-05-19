# Contrat de Donnees FootNet

Placez ici vos exports reels si vous voulez que FootNetViz les charge automatiquement.

Formats supportes:

- `matches.csv`, `matches.json`, `matches.parquet`
- `players.csv`, `players.json`, `players.parquet`
- `passes.csv`, `passes.json`, `passes.parquet`
- `roster.csv`, `roster.json`, `roster.parquet` optionnel

## Tables minimales

### `matches`

Colonnes requises:

- `team`
- `opponent`
- `match_day` ou `match_id`

Colonnes utiles acceptees:

- `xg_for`, `xg_against`
- `possession`
- `pass_accuracy`
- `press_intensity`
- `final_third_entries`
- `scoreline`
- `score_for`, `score_against`

Alias compris notamment:

- `team_name` -> `team`
- `opponent_name` -> `opponent`
- `matchday` -> `match_day`
- `fixture_id`, `game_id` -> `match_id`

### `players`

Colonnes requises:

- `team`
- `player`
- `match_day` ou `match_id`

Colonnes utiles acceptees:

- `role`
- `minutes`
- `touches`
- `progressive_passes`
- `shot_assists`
- `xT_added`
- `duels_won`
- `x`, `y`

Alias compris notamment:

- `player_name` -> `player`
- `position` -> `role`
- `prog_passes` -> `progressive_passes`
- `key_passes` -> `shot_assists`
- `xt` -> `xT_added`

### `passes`

Colonnes requises:

- `team`
- `passer`
- `receiver`
- `count`
- `match_day` ou `match_id`

Colonnes utiles acceptees:

- `passer_role`, `receiver_role`
- `passer_x`, `passer_y`
- `receiver_x`, `receiver_y`

Alias compris notamment:

- `from` -> `passer`
- `to` -> `receiver`
- `pass_count` -> `count`

### `roster`

Optionnel. Si absent, il est reconstruit depuis `players`.

Colonnes utiles:

- `team`
- `player`
- `role`
- `x`, `y`

## Notes pratiques

- Si `players` et `passes` n'ont qu'un `match_id`, FootNetViz reconstruit `match_day` depuis `matches`.
- Si les positions `x` et `y` manquent, l'application applique un placement par role.
- Si les donnees reelles sont invalides ou incompletes, l'application repasse en mode demo.
- Pour utiliser un dossier externe au repo, definissez `FOOTNET_DATA_DIR` avant de lancer Streamlit.