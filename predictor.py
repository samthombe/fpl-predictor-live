# predictor.py -- SAFE CLOUD VERSION
import pandas as pd
import numpy as np
from pathlib import Path

# Try importing joblib, but don't crash if it's missing
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("WARNING: joblib not available. Using heuristics only.")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODELS_LOADED = False
CALIBRATORS_LOADED = False
MODEL_MEAN = None
MODEL_Q10 = None
MODEL_Q90 = None
CALIBRATORS = None

if JOBLIB_AVAILABLE and MODEL_DIR.exists():
    try:
        # Check for bundled quantiles first
        if (MODEL_DIR / "lgb_quantiles.pkl").exists():
            QUANTILE_MODELS = joblib.load(MODEL_DIR / "lgb_quantiles.pkl")
            MODEL_MEAN = QUANTILE_MODELS.get('mean') or QUANTILE_MODELS.get('q50')
            MODEL_Q10 = QUANTILE_MODELS.get('q10')
            MODEL_Q90 = QUANTILE_MODELS.get('q90')
            MODELS_LOADED = True
            print("INFO: Loaded bundled quantile models.")
        
        # Fallback to individual pkl files
        elif (MODEL_DIR / "lgb_mean.pkl").exists():
            MODEL_MEAN = joblib.load(MODEL_DIR / "lgb_mean.pkl")
            MODEL_Q10 = joblib.load(MODEL_DIR / "lgb_q10.pkl") if (MODEL_DIR / "lgb_q10.pkl").exists() else None
            MODEL_Q90 = joblib.load(MODEL_DIR / "lgb_q90.pkl") if (MODEL_DIR / "lgb_q90.pkl").exists() else None
            MODELS_LOADED = True
            print("INFO: Loaded individual model files.")
            
        # Load calibrators if present
        if (MODEL_DIR / "isotonic_calibrators.pkl").exists():
            CALIBRATORS = joblib.load(MODEL_DIR / "isotonic_calibrators.pkl")
            CALIBRATORS_LOADED = True
            
    except Exception as e:
        print(f"ERROR loading models: {e}. Falling back to heuristics.")
        MODELS_LOADED = False
else:
    print("INFO: No models found or joblib unavailable. Using heuristics.")

# ... rest of your predictor.py code remains unchanged ...
# Ensure _heuristic_projection is defined below this block so it works even if MODELS_LOADED is False


FEATURE_COLS = [
    "position", "team_code", "gw", "is_home", "fdr", "opp_code",
    "form_1", "form_3", "form_5", "ppg_lag",
    "xg_lag1", "xa_lag1", "def_actions_sum_lag1",
    "minutes_lag1", "starts_lag1", "goals_lag1", "assists_lag1", 
    "bonus_lag1", "clean_sheets_lag1",
    "price_lag1", "sel_lag1",
]

POS_MAP = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}


def _heuristic_projection(p):
    recent = p["form"].astype(float).clip(upper=8.0)
    season = p["points_per_game"].astype(float).clip(upper=12.0)
    base = 0.45 * recent + 0.55 * season
    xgi = p["expected_goals"].astype(float) + p["expected_assists"].astype(float)
    underlying = pd.Series(np.minimum(1.5, 0.08 * xgi), index=p.index)
    defensive = pd.Series(
        np.where(p["position"].isin(["GK", "DEF"]), np.minimum(0.8, 0.05 * p["clean_sheets"].astype(float)), 0.0),
        index=p.index,
    )
    raw = base + underlying + defensive
    return (raw * p["fixture_factor"] * p["chance_factor"]).clip(lower=0.0)


def detect_dgw(fixtures_list):
    """Returns dict: team_code -> list of GWs with >1 fixture."""
    if not fixtures_list:
        return {}
    team_gws = {}
    for f in fixtures_list:
        try:
            h = int(f.get("team_h_code"))
            a = int(f.get("team_a_code"))
            gw = int(f.get("event"))
            team_gws.setdefault(h, []).append(gw)
            team_gws.setdefault(a, []).append(gw)
        except Exception:
            continue
    
    dgw_teams = {}
    for team, gws in team_gws.items():
        unique_gws = set(gws)
        if len(gws) > len(unique_gws): # Duplicate GW found
            dgw_teams[team] = [g for g in unique_gws if gws.count(g) > 1]
    return dgw_teams


def apply_strategic_layers(p, dgw_map=None):
    """Post-process ML outputs to fix overconfidence and add fixture awareness."""
    
    # --- Layer 1: Widen intervals IF NO CALIBRATOR EXISTS ---
    if not CALIBRATORS_LOADED:
        expansion_factor = 1.10
        mean = p["projected_points"]
        floor = p["predicted_floor"]
        ceiling = p["predicted_ceiling"]
        
        new_floor = mean - (mean - floor) * expansion_factor
        new_ceiling = mean + (ceiling - mean) * expansion_factor
        
        p["predicted_floor"] = np.maximum(new_floor, 0.0)
        p["predicted_ceiling"] = np.minimum(new_ceiling, 20.0)
    
    # --- Layer 2: Fixture-aware adjustment ---
    fdr_adjustment = pd.Series(1.0, index=p.index)
    if "fdr" in p.columns:
        fdr_vals = pd.to_numeric(p["fdr"], errors="coerce").fillna(3)
        fdr_adjustment = np.where(fdr_vals <= 2, 1.05,   # +5% for very easy fixtures
                         np.where(fdr_vals >= 4, 0.95,   # -5% for very hard fixtures
                                  1.0))                   # neutral otherwise
    
    # --- Layer 3: DGW Boost ---
    if dgw_map:
        for team_code, dgs in dgw_map.items():
            mask = p["team_code"] == team_code
            if mask.any():
                # Boost projections for teams with double gameweeks
                boost = 1.15 
                p.loc[mask, "projected_points"] *= boost
                p.loc[mask, "predicted_floor"] *= boost
                p.loc[mask, "predicted_ceiling"] *= boost

    p["projected_points"] *= fdr_adjustment
    p["predicted_floor"] *= fdr_adjustment
    p["predicted_ceiling"] *= fdr_adjustment
    
    # Recalculate value after adjustments
    p["value"] = (p["projected_points"] / p["price_m"].replace(0, np.nan)).fillna(0.0)
    
    return p


def project_players(players: pd.DataFrame, teams: pd.DataFrame, fixtures) -> pd.DataFrame:
    """Projects points using ML models if available, else falls back to heuristics."""
    p = players.copy()
    if p.empty:
        return p

    # --- Compute Fixture Factors ---
    fixture_factors = {}
    if fixtures:
        for f in fixtures:
            try:
                h = int(f.get("team_h_code"))
                a = int(f.get("team_a_code"))
                hd = f.get("team_h_difficulty", 3)
                ad = f.get("team_a_difficulty", 3)
                def diff_to_factor(d):
                    d = max(1, min(5, int(d)))
                    return float(np.clip(1.18 - 0.07 * (d - 1), 0.85, 1.20))
                fixture_factors[h] = diff_to_factor(hd)
                fixture_factors[a] = diff_to_factor(ad)
            except Exception:
                continue
    
    p["fixture_factor"] = p["team_code"].map(fixture_factors).fillna(1.0)
    p["chance_factor"] = 0.40 + 0.60 * (p["chance_pct"].astype(float) / 100.0)

    # --- Detect DGWs INSIDE FUNCTION ---
    dgw_map = detect_dgw(fixtures)

    # --- Prepare Features for ML ---
    if MODELS_LOADED:
        p["pos_int"] = p["position"].map(POS_MAP).fillna(0).astype(int)
        
        # Approximate lagged features from current API stats
        p["form_1"] = p["event_points"]       
        p["form_3"] = p["form"]               
        p["form_5"] = p["points_per_game"]    
        p["ppg_lag"] = p["points_per_game"]
        
        p["xg_lag1"] = p["expected_goals"]
        p["xa_lag1"] = p["expected_assists"]
        p["def_actions_sum_lag1"] = 0.0       
        
        p["minutes_lag1"] = p["minutes"]
        p["starts_lag1"] = p["starts"]
        p["goals_lag1"] = p["goals_scored"]
        p["assists_lag1"] = p["assists"]
        p["bonus_lag1"] = p["bonus"]
        p["clean_sheets_lag1"] = p["clean_sheets"]
        
        p["price_lag1"] = p["price_m"]
        p["sel_lag1"] = p["selected_by_percent"]
        
        # Derive FDR from fixture_factor
        def factor_to_fdr(x):
            if x >= 1.15: return 1
            elif x >= 1.05: return 2
            elif x >= 0.95: return 3
            elif x >= 0.85: return 4
            else: return 5
        p["fdr"] = p["fixture_factor"].apply(factor_to_fdr)
        
        p["is_home"] = 0  
        p["opp_code"] = 0 
        
        X_live = p[[c for c in FEATURE_COLS if c in p.columns]].copy()
        
        try:
            pred_mean = MODEL_MEAN.predict(X_live)
            
            # Use Q10/Q90 if available
            if MODEL_Q10 is not None:
                pred_q10 = MODEL_Q10.predict(X_live)
            else:
                pred_q10 = pred_mean * 0.7 # Rough fallback
                
            if MODEL_Q90 is not None:
                pred_q90 = MODEL_Q90.predict(X_live)
            else:
                pred_q90 = pred_mean * 1.3 # Rough fallback
            
            p["projected_points"] = pred_mean
            p["predicted_floor"] = pred_q10
            p["predicted_ceiling"] = pred_q90
            
            # Apply chance adjustment lightly
            p["projected_points"] *= p["chance_factor"]
            p["predicted_floor"] *= p["chance_factor"]
            p["predicted_ceiling"] *= p["chance_factor"]
            
            # Apply strategic layers (includes DGW boost)
            p = apply_strategic_layers(p, dgw_map=dgw_map)
            
        except Exception as e:
            print(f"ML Prediction failed: {e}. Using heuristics.")
            p["projected_points"] = _heuristic_projection(p)
            p["predicted_floor"] = p["projected_points"] * 0.7
            p["predicted_ceiling"] = p["projected_points"] * 1.3
            # Still apply DGW/FDR boosts even on heuristic fallback
            p = apply_strategic_layers(p, dgw_map=dgw_map)
    else:
        p["projected_points"] = _heuristic_projection(p)
        p["predicted_floor"] = p["projected_points"] * 0.7
        p["predicted_ceiling"] = p["projected_points"] * 1.3
        p = apply_strategic_layers(p, dgw_map=dgw_map)

    p["value"] = (p["projected_points"] / p["price_m"].replace(0, np.nan)).fillna(0.0)
    
    return p.sort_values("projected_points", ascending=False).reset_index(drop=True)