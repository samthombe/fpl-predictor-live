import requests
import pandas as pd
import numpy as np

BASE_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURE_URL = "https://fantasy.premierleague.com/api/fixtures/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://fantasy.premierleague.com/",
}

NUMERIC_DEFAULTS = [
    "form", "points_per_game", "total_points", "event_points",
    "expected_goals", "expected_assists", "clean_sheets",
    "goals_scored", "assists", "bonus", "influence", "creativity",
    "threat", "ict_index", "starts", "minutes", "selected_by_percent",
    "now_cost", "element_type", "team_code", "id",
]

HARD_POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
VALID_POS = {"GK", "DEF", "MID", "FWD"}


def _get_json(url, params=None):
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _ensure_numeric_defaults(p: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_DEFAULTS:
        if col not in p.columns:
            p[col] = 0
        p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0)
    return p


def _build_names(p: pd.DataFrame) -> pd.DataFrame:
    if "web_name" not in p.columns or p["web_name"].isna().all():
        if "first_name" in p.columns and "second_name" in p.columns:
            p["web_name"] = (
                p["first_name"].astype(str).fillna("") + " " +
                p["second_name"].astype(str).fillna("")
            ).str.strip()
        elif "first_name" in p.columns:
            p["web_name"] = p["first_name"].astype(str)
        else:
            p["web_name"] = "?"
    p["web_name"] = p["web_name"].astype(str).replace({"": "?", "nan": "?"})
    return p


def _build_position_map(element_types) -> dict:
    m = dict(HARD_POS)
    if not element_types:
        return m
    for et in element_types:
        try:
            eid = int(et.get("id"))
            short = str(et.get("singular_name_short", "")).upper()
            if short in VALID_POS:
                m[eid] = short
        except Exception:
            continue
    return m


def _select_next_fixtures(fixtures, events_df):
    """Return (fixtures_for_next_gw, next_gw_id). Falls back safely."""
    if not fixtures:
        return fixtures, None
    try:
        next_gw = None
        if events_df is not None and not events_df.empty and "id" in events_df.columns:
            ev = events_df.copy()
            ev["id"] = pd.to_numeric(ev["id"], errors="coerce")
            ev = ev.dropna(subset=["id"])
            if "finished" in ev.columns:
                fin = ev["finished"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
                not_fin = ev[~fin]
                if not not_fin.empty:
                    next_gw = int(not_fin["id"].min())
                else:
                    fin_ids = ev.loc[fin, "id"]
                    next_gw = int(fin_ids.max()) + 1 if not fin_ids.empty else int(ev["id"].min())
            else:
                next_gw = int(ev["id"].min())
        if next_gw is not None:
            filtered = [f for f in fixtures if f.get("event") == next_gw]
            if filtered:
                return filtered, next_gw
        return fixtures, next_gw
    except Exception:
        return fixtures, None


def load_raw_data():
    """Returns (players, teams, events, fixtures, debug, element_types)."""
    bootstrap = _get_json(BASE_URL)

    debug = {
        "top_level_keys": list(bootstrap.keys()) if isinstance(bootstrap, dict) else type(bootstrap).__name__,
        "raw_player_count": 0,
        "player_columns_sample": [],
        "status_value_counts": {},
        "has_chance_col": False,
        "has_now_cost": False,
        "fixtures_ok": False,
        "next_gw": None,
        "fixtures_used": 0,
        "sample_element": None,
        "error": None,
    }

    # *** THE FIX: the array is called "elements", not "players" ***
    players_raw = bootstrap.get("elements", []) if isinstance(bootstrap, dict) else []
    teams_raw = bootstrap.get("teams", []) if isinstance(bootstrap, dict) else []
    events_raw = bootstrap.get("events", []) if isinstance(bootstrap, dict) else []
    element_types = bootstrap.get("element_types", []) if isinstance(bootstrap, dict) else []

    debug["raw_player_count"] = len(players_raw)
    if players_raw:
        first = players_raw[0]
        debug["player_columns_sample"] = sorted(list(first.keys()))
        debug["has_chance_col"] = "chance_of_playing_next_round" in first
        debug["has_now_cost"] = "now_cost" in first
        # keep a small trimmed sample so we can verify field names if anything is off
        debug["sample_element"] = {
            k: first.get(k)
            for k in ["id", "web_name", "first_name", "second_name", "team_code",
                      "element_type", "status", "now_cost", "form", "points_per_game",
                      "expected_goals", "expected_assists", "selected_by_percent"]
            if k in first
        }

    players = pd.DataFrame(players_raw)
    teams = pd.DataFrame(teams_raw)
    events = pd.DataFrame(events_raw)

    try:
        fixtures = _get_json(FIXTURE_URL)
        debug["fixtures_ok"] = isinstance(fixtures, list)
        if not isinstance(fixtures, list):
            fixtures = []
    except Exception:
        fixtures = []

    fixtures, next_gw = _select_next_fixtures(fixtures, events)
    debug["next_gw"] = next_gw
    debug["fixtures_used"] = len(fixtures)

    if not players.empty and "status" in players.columns:
        debug["status_value_counts"] = (
            players["status"].astype(str).str.lower().value_counts().to_dict()
        )

    return players, teams, events, fixtures, debug, element_types


def clean_players(players: pd.DataFrame, teams: pd.DataFrame, position_map: dict) -> pd.DataFrame:
    p = players.copy()
    if p.empty:
        return p

    p = _ensure_numeric_defaults(p)
    p = _build_names(p)

    p["position"] = pd.to_numeric(p["element_type"], errors="coerce").map(position_map)

    team_map = {}
    if not teams.empty and "code" in teams.columns:
        t = teams.copy()
        t["code"] = pd.to_numeric(t["code"], errors="coerce")
        t = t.dropna(subset=["code"]).astype({"code": int})
        name_col = "short_name" if "short_name" in t.columns else ("name" if "name" in t.columns else None)
        if name_col:
            team_map = dict(zip(t["code"], t[name_col].astype(str)))
    p["team"] = p["team_code"].astype(int).map(team_map).fillna("?")

    # now_cost is in tenths of £m (55 -> £5.5m)
    p["price_m"] = pd.to_numeric(p["now_cost"], errors="coerce").fillna(0) / 10.0

    if "status" in p.columns:
        p["status"] = p["status"].astype(str).str.lower()
    else:
        p["status"] = "a"

    if "chance_of_playing_next_round" in p.columns:
        chance = pd.to_numeric(p["chance_of_playing_next_round"], errors="coerce")
        p["chance_pct"] = chance.fillna(100).clip(lower=0, upper=100)
    else:
        p["chance_pct"] = 100.0

    strict = p[
        p["status"].isin(["a", "d"]) &
        (p["chance_pct"] >= 25) &
        (p["price_m"] > 0)
    ].copy()

    if len(strict) == 0 and len(p) > 0:
        strict = p[p["price_m"] > 0].copy()
        strict["_relaxed"] = True

    strict = strict[strict["position"].isin(list(VALID_POS))]
    return strict.reset_index(drop=True)


def load_clean_data():
    players, teams, events, fixtures, debug, element_types = load_raw_data()
    position_map = _build_position_map(element_types)
    cleaned = clean_players(players, teams, position_map)
    return cleaned, teams, events, fixtures, debug