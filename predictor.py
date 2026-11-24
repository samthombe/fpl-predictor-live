# predictor.py -- ALIGNED FEATURES + FIXTURE MAPPING + VISIBLE DEBUG
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_MEAN = MODEL_Q10 = MODEL_Q90 = None
MODELS_LOADED = False
LAST_DEBUG = {"engine": "heuristic", "models_loaded": False}

if JOBLIB_OK:
    try:
        if (MODEL_DIR / "lgb_quantiles.pkl").exists():
            m = joblib.load(MODEL_DIR / "lgb_quantiles.pkl")
            MODEL_MEAN = m.get("mean") or m.get("q50")
            MODEL_Q10, MODEL_Q90 = m.get("q10"), m.get("q90")
        elif (MODEL_DIR / "lgb_mean.pkl").exists():
            MODEL_MEAN = joblib.load(MODEL_DIR / "lgb_mean.pkl")
            if (MODEL_DIR / "lgb_q10.pkl").exists():
                MODEL_Q10 = joblib.load(MODEL_DIR / "lgb_q10.pkl")
            if (MODEL_DIR / "lgb_q90.pkl").exists():
                MODEL_Q90 = joblib.load(MODEL_DIR / "lgb_q90.pkl")
        MODELS_LOADED = MODEL_MEAN is not None
    except Exception as e:
        LAST_DEBUG["model_load_error"] = str(e)

LAST_DEBUG["models_loaded"] = MODELS_LOADED

# MUST match train_model.py FEATURE_COLS exactly (19 features).
FEATURE_COLS = [
    "position", "gw", "is_home", "fdr",
    "form_1", "form_3", "form_5", "ppg_lag",
    "xg_lag1", "xa_lag1", "def_actions_sum_lag1",
    "minutes_lag1", "starts_lag1", "goals_lag1", "assists_lag1",
    "bonus_lag1", "clean_sheets_lag1",
    "price_lag1", "sel_lag1",
]
POS_MAP = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}


def _fixture_context(teams, fixtures):
    """Map fixture team IDs -> codes, then build per-team fixture summary."""
    id_to_code = {}
    if teams is not None and not teams.empty and {"id", "code"} <= set(teams.columns):
        t = teams[["id", "code"]].apply(pd.to_numeric, errors="coerce").dropna()
        id_to_code = {int(r.id): int(r.code) for r in t.itertuples(index=False)}

    per_team = {}
    for f in fixtures or []:
        try:
            h = id_to_code.get(int(f.get("team_h")))
            a = id_to_code.get(int(f.get("team_a")))
        except (TypeError, ValueError):
            continue
        if h is None or a is None:
            continue
        hd = int(f.get("team_h_difficulty") or 3)
        ad = int(f.get("team_a_difficulty") or 3)
        per_team.setdefault(h, []).append((hd, 1, a))
        per_team.setdefault(a, []).append((ad, 0, h))

    summary = {}
    for code, games in per_team.items():
        fdr = float(np.mean([g[0] for g in games]))
        summary[code] = {
            "fdr": fdr,
            "is_home": int(any(g[1] for g in games)),
            "opp": games[0][2],
            "n": len(games),
            "factor": float(np.clip(1.18 - 0.07 * (fdr - 1), 0.85, 1.20)),
        }
    return summary


def _heuristic_projection(p):
    recent = pd.to_numeric(p["form"], errors="coerce").fillna(0).clip(upper=8.0)
    season = pd.to_numeric(p["points_per_game"], errors="coerce").fillna(0).clip(upper=12.0)
    base = 0.45 * recent + 0.55 * season
    xgi = (pd.to_numeric(p["expected_goals"], errors="coerce").fillna(0)
           + pd.to_numeric(p["expected_assists"], errors="coerce").fillna(0))
    underlying = np.minimum(1.5, 0.08 * xgi)
    defensive = np.where(
        p["position"].isin(["GK", "DEF"]),
        np.minimum(0.8, 0.05 * pd.to_numeric(p["clean_sheets"], errors="coerce").fillna(0)),
        0.0,
    )
    raw = base + underlying + defensive
    return (raw * p["fixture_factor"] * p["chance_factor"]).clip(lower=0.0)


def project_players(players: pd.DataFrame, teams: pd.DataFrame, fixtures) -> pd.DataFrame:
    p = players.copy()
    if p.empty:
        return p

    p["team_code"] = pd.to_numeric(p["team_code"], errors="coerce").fillna(0).astype(int)

    fx = _fixture_context(teams, fixtures)
    p["fixture_factor"] = p["team_code"].map(lambda c: fx.get(c, {}).get("factor", 1.0))
    p["fdr"] = p["team_code"].map(lambda c: fx.get(c, {}).get("fdr", 3.0))
    p["is_home"] = p["team_code"].map(lambda c: fx.get(c, {}).get("is_home", 0))
    p["opp_code"] = p["team_code"].map(lambda c: fx.get(c, {}).get("opp", 0)).astype(int)
    p["n_fixtures"] = p["team_code"].map(lambda c: fx.get(c, {}).get("n", 0)).astype(int)

    p["chance_factor"] = 0.40 + 0.60 * (
        pd.to_numeric(p["chance_pct"], errors="coerce").fillna(100) / 100.0
    )

    gw = 0
    if fixtures:
        try:
            gw = int(fixtures[0].get("event") or 0)
        except (TypeError, ValueError):
            gw = 0

    LAST_DEBUG.update({
        "fixtures_received": len(fixtures or []),
        "teams_with_fixture": len(fx),
        "players_matched": int((p["fixture_factor"] != 1.0).sum()),
        "gw_used": gw,
        "features_used": len(FEATURE_COLS),
        "sample_fixture_keys": sorted(list((fixtures or [{}])[0].keys()))[:12],
    })

    used_ml = False
    if MODELS_LOADED:
        try:
            # Exact training columns, correct dtypes.
            # Counting stats are season-to-date totals -- matches the ALIGNED builder.
            X = pd.DataFrame(index=p.index)
            X["position"] = p["position"].map(POS_MAP).fillna(0).astype(int)
            X["gw"] = gw
            X["is_home"] = p["is_home"].astype(int)
            X["fdr"] = p["fdr"].astype(float)
            X["form_1"] = pd.to_numeric(p["event_points"], errors="coerce")
            X["form_3"] = pd.to_numeric(p["form"], errors="coerce")
            X["form_5"] = pd.to_numeric(p["points_per_game"], errors="coerce")
            X["ppg_lag"] = X["form_5"]
            X["xg_lag1"] = pd.to_numeric(p["expected_goals"], errors="coerce")
            X["xa_lag1"] = pd.to_numeric(p["expected_assists"], errors="coerce")
            X["def_actions_sum_lag1"] = np.nan  # not available live; LGB handles NaN
            X["minutes_lag1"] = pd.to_numeric(p["minutes"], errors="coerce")
            X["starts_lag1"] = pd.to_numeric(p["starts"], errors="coerce")
            X["goals_lag1"] = pd.to_numeric(p["goals_scored"], errors="coerce")
            X["assists_lag1"] = pd.to_numeric(p["assists"], errors="coerce")
            X["bonus_lag1"] = pd.to_numeric(p["bonus"], errors="coerce")
            X["clean_sheets_lag1"] = pd.to_numeric(p["clean_sheets"], errors="coerce")
            X["price_lag1"] = pd.to_numeric(p["price_m"], errors="coerce")
            X["sel_lag1"] = pd.to_numeric(p["selected_by_percent"], errors="coerce")
            X = X[FEATURE_COLS]

            proj = np.asarray(MODEL_MEAN.predict(X), dtype=float)
            floor = (np.asarray(MODEL_Q10.predict(X), dtype=float)
                     if MODEL_Q10 is not None else proj * 0.7)
            ceil = (np.asarray(MODEL_Q90.predict(X), dtype=float)
                    if MODEL_Q90 is not None else proj * 1.3)

            cf = p["chance_factor"].to_numpy()
            proj, floor, ceil = proj * cf, floor * cf, ceil * cf

            # Widen intervals (measured coverage 69% vs 80% target).
            floor = np.maximum(proj - (proj - floor) * 1.10, 0.0)
            ceil = proj + (ceil - proj) * 1.10

            # Enforce ordering: floor <= mean <= ceiling (fixes quantile crossing).
            floor = np.minimum(floor, proj)
            ceil = np.maximum(ceil, proj)

            p["projected_points"] = proj
            p["predicted_floor"] = floor
            p["predicted_ceiling"] = ceil
            used_ml = True
            LAST_DEBUG["engine"] = "ml"
            LAST_DEBUG.pop("ml_error", None)
        except Exception as e:
            LAST_DEBUG["engine"] = "heuristic"
            LAST_DEBUG["ml_error"] = f"{type(e).__name__}: {e}"

    if not used_ml:
        base = _heuristic_projection(p)
        p["projected_points"] = base
        p["predicted_floor"] = base * 0.7
        p["predicted_ceiling"] = base * 1.3

    # DGW boost (shared): ~1.75x for two games, allowing for rotation risk.
    n = p["n_fixtures"].to_numpy()
    boost = 1.0 + 0.75 * np.maximum(n - 1, 0)
    p["projected_points"] = p["projected_points"] * boost
    p["predicted_floor"] = p["predicted_floor"] * boost
    p["predicted_ceiling"] = np.minimum(p["predicted_ceiling"] * boost, 20.0 * np.maximum(n, 1))

    # Final safety: keep floor <= projected <= ceiling after all adjustments.
    p["predicted_floor"] = np.minimum(p["predicted_floor"], p["projected_points"])
    p["predicted_ceiling"] = np.maximum(p["predicted_ceiling"], p["projected_points"])

    p["value"] = (p["projected_points"] / p["price_m"].replace(0, np.nan)).fillna(0.0)
    return p.sort_values("projected_points", ascending=False).reset_index(drop=True)