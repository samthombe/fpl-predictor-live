# Track B / Part 1 -- FINAL VERSION.
# Strategy: Stack gws/gw*.csv files directly. No more guessing filenames or traversing 
# 600+ player folders. Uses observed structure from probe_tree.py.
# Features: Extracts xG, xA, defensive actions, penalties, etc., for a high-signal ML model.

import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

BASE_DIR = Path(__file__).resolve().parent
HISTORY_DIR = BASE_DIR / "data" / "vaastav" / "data"  # Note: vaastav has nested 'data' folder
OUT_DIR = BASE_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "training_set.csv"

SEASONS_TO_USE = 3  # Last 3 seasons

# Mapping of source column names to our canonical feature names
# Source cols come from gws/gwX.csv headers seen in probe output
COL_MAP = {
    "player_id": ["element"],       # 'element' is the FPL API ID
    "gw": ["round"],                # 'round' is the GW number
    "y": ["total_points"],          # Label: Actual points scored this GW
    
    # Core Stats
    "minutes": ["minutes"],
    "starts": ["starts"],
    "goals": ["goals_scored"],
    "assists": ["assists"],
    "bonus": ["bonus"],
    "clean_sheets": ["clean_sheets"],
    
    # Advanced Underlying Metrics (The Edge)
    "xg": ["expected_goals"],
    "xa": ["expected_assists"],
    "xgi": ["expected_goal_involvements"],
    "def_actions": ["defensive_contribution", "tackles", "interceptions", "blocks", "clearances"], # Sum these later
    "pens_missed": ["penalties_missed"],
    "pens_saved": ["penalties_saved"],
    "saves": ["saves"],
    "goals_conceded": ["goals_conceded"],
    
    # Context
    "team_code": ["team_a_code", "team_h_code"], # Need logic to determine which is player's team
    "opponent_team": ["opponent_team"],
    "is_home": ["was_home"],
    "fdr": [], # Will derive from fixtures or default if missing
    
    # Market/Selection
    "sel_pct": ["selected"],        # Often percentage or count depending on version
    "price_m": ["value"],           # 'value' in gw files is usually current market value (£m)
}

POS_INT = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

def log(msg):
    print(msg, flush=True)

def discover_seasons():
    """Find season folders under data/vaastav/data/"""
    if not HISTORY_DIR.exists():
        raise FileNotFoundError(f"History dir not found: {HISTORY_DIR}")
    
    pat = re.compile(r"20\d\d-\d\d")
    dirs = [p for p in HISTORY_DIR.iterdir() if p.is_dir() and pat.search(p.name)]
    dirs.sort(key=lambda d: d.name, reverse=True)
    
    log(f"Discovered {len(dirs)} season(s): {[d.name for d in dirs]}")
    return dirs[:SEASONS_TO_USE]

def load_gw_files(season_dir):
    """Load and stack all gws/gw*.csv files for a season."""
    gws_dir = season_dir / "gws"
    if not gws_dir.exists():
        log(f"  [{season_dir.name}] No 'gws/' folder found.")
        return None
    
    csvs = sorted(gws_dir.glob("gw*.csv"))
    if not csvs:
        log(f"  [{season_dir.name}] No gw*.csv files in 'gws/'.")
        return None
        
    log(f"  [{season_dir.name}] Found {len(csvs)} GW files. Stacking...")
    
    frames = []
    for f in csvs:
        try:
            df = pd.read_csv(f, low_memory=False)
            # Inject filename-derived GW if 'round' col is missing/weird
            if "round" not in df.columns:
                m = re.search(r"gw(\d+)\.csv", f.name.lower())
                if m:
                    df["round"] = int(m.group(1))
            frames.append(df)
        except Exception as e:
            log(f"    ! Failed reading {f.name}: {e}")
            
    if not frames:
        return None
        
    return pd.concat(frames, ignore_index=True)

def process_season(season_dir):
    name = season_dir.name
    log(f"Processing season: {name}")
    
    raw_df = load_gw_files(season_dir)
    if raw_df is None or raw_df.empty:
        log(f"  [{name}] SKIP: No data loaded.")
        return None
        
    # --- Schema Resolution & Cleaning ---
    out = pd.DataFrame()
    
    # 1. IDs & Labels
    out["player_id"] = pd.to_numeric(raw_df.get("element"), errors="coerce")
    out["gw"] = pd.to_numeric(raw_df.get("round"), errors="coerce")
    out["y"] = pd.to_numeric(raw_df.get("total_points"), errors="coerce")
    out["season"] = name
    
    # Drop rows with missing essential keys
    out = out.dropna(subset=["player_id", "gw", "y"])
    out["player_id"] = out["player_id"].astype(int)
    out["gw"] = out["gw"].astype(int)
    
    # 2. Position (Need to map element_type if available, else infer or drop)
    # Vaastav gw files might not have element_type directly, check columns
    pos_col = None
    for c in ["element_type", "position", "squad_position"]:
        if c in raw_df.columns:
            pos_col = c
            break
            
    if pos_col:
        out["position"] = pd.to_numeric(raw_df[pos_col], errors="coerce").map(POS_INT)
    else:
        # Fallback: If no position info, we can't filter by pos later. 
        # For MVP, we might assume MID/FWD bias or just leave NaN. 
        # Better: Try to join with teams.csv or cleaned_players.csv if needed, 
        # but for speed, let's proceed with NaN and handle in optimizer/app.
        out["position"] = np.nan
        log(f"  [{name}] WARNING: No position column found in gw files.")

    # 3. Team Code Logic
    # In gw files, 'team_h_code' and 'team_a_code' exist. 
    # We need to know WHICH team the player belongs to.
    # Usually, there's a 'team_code' column or we infer from 'element' joining with master list.
    # Let's look for explicit team code first.
    tc_col = None
    for c in ["team_code", "team"]:
        if c in raw_df.columns:
            tc_col = c
            break
            
    if tc_col:
        out["team_code"] = pd.to_numeric(raw_df[tc_col], errors="coerce")
    else:
        # Infer: If player was home, their team is team_h_code? No, that's unsafe.
        # Safest bet: Join with 'cleaned_players.csv' or 'player_idlist.csv' in same season dir.
        id_list_path = season_dir / "player_idlist.csv"
        if id_list_path.exists():
            ids = pd.read_csv(id_list_path)
            # Map id -> team_code (assuming 'code' or 'team_code' exists in idlist)
            t_col = "team_code" if "team_code" in ids.columns else ("code" if "code" in ids.columns else None)
            if t_col:
                mapper = dict(zip(ids["id"].astype(int), ids[t_col].astype(int)))
                out["team_code"] = out["player_id"].map(mapper)
            else:
                out["team_code"] = np.nan
        else:
            out["team_code"] = np.nan
            log(f"  [{name}] WARNING: Could not resolve team codes.")

    # 4. Numeric Features
    num_cols = [
        "minutes", "starts", "goals_scored", "assists", "bonus", "clean_sheets",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "penalties_missed", "penalties_saved", "saves", "goals_conceded",
        "tackles", "interceptions", "blocks", "clearances", "defensive_contribution",
        "value", "selected"
    ]
    
    for col in num_cols:
        if col in raw_df.columns:
            out[col] = pd.to_numeric(raw_df[col], errors="coerce")
        else:
            out[col] = np.nan
            
    # Rename for consistency with downstream expectations
    rename_map = {
        "expected_goals": "xg",
        "expected_assists": "xa",
        "expected_goal_involvements": "xgi",
        "goals_scored": "goals",
        "value": "price_m", # Assuming value is £m
        "selected": "sel_count" # Might be %, check range later
    }
    out = out.rename(columns=rename_map)
    
    # 5. Derived Defensive Action Score
    # Sum tackles, ints, blocks, clears if individual cols exist, else use defensive_contribution
    def_cols = ["tackles", "interceptions", "blocks", "clearances"]
    existing_def = [c for c in def_cols if c in out.columns and out[c].notna().any()]
    
    if existing_def:
        out["def_actions_sum"] = out[existing_def].sum(axis=1)
    elif "defensive_contribution" in out.columns:
        out["def_actions_sum"] = out["defensive_contribution"]
    else:
        out["def_actions_sum"] = np.nan
        
    # 6. Home/Away & Opponent
    if "was_home" in raw_df.columns:
        out["is_home"] = pd.to_numeric(raw_df["was_home"], errors="coerce")
    else:
        out["is_home"] = np.nan
        
    if "opponent_team" in raw_df.columns:
        out["opp_code"] = pd.to_numeric(raw_df["opponent_team"], errors="coerce")
    else:
        out["opp_code"] = np.nan

    # 7. Fixture Difficulty (FDR)
    # Try loading fixtures.csv from same season dir
    fix_path = season_dir / "fixtures.csv"
    if fix_path.exists():
        try:
            fx = pd.read_csv(fix_path)
            # Resolve columns
            gw_c = next((c for c in fx.columns if c.lower() in ["round", "event"]), None)
            h_c = next((c for c in fx.columns if c.lower() in ["team_h_code", "home_team"]), None)
            a_c = next((c for c in fx.columns if c.lower() in ["team_a_code", "away_team"]), None)
            hd_c = next((c for c in fx.columns if "diff" in c.lower() and "h" in c.lower()), None)
            ad_c = next((c for c in fx.columns if "diff" in c.lower() and "a" in c.lower()), None)
            
            if gw_c and h_c and a_c:
                # Create lookup: (gw, team_code) -> fdr
                fx_rows = []
                for _, r in fx.iterrows():
                    g = int(r[gw_c])
                    h = int(r[h_c])
                    a = int(r[a_c])
                    hd = float(r[hd_c]) if hd_c and not pd.isna(r[hd_c]) else 3.0
                    ad = float(r[ad_c]) if ad_c and not pd.isna(r[ad_c]) else 3.0
                    fx_rows.append((g, h, hd))
                    fx_rows.append((g, a, ad))
                
                fdr_lookup = {(g, t): d for g, t, d in fx_rows}
                out["fdr"] = out.apply(lambda row: fdr_lookup.get((row["gw"], row["team_code"]), 3.0), axis=1)
            else:
                out["fdr"] = 3.0
        except Exception as e:
            log(f"  [{name}] FDR calc failed: {e}. Defaulting to 3.0.")
            out["fdr"] = 3.0
    else:
        out["fdr"] = 3.0

    # Final Cleanup
    out = out.drop_duplicates(subset=["season", "player_id", "gw"])
    
    log(f"  [{name}] Processed {len(out)} rows.")
    return out

def add_lagged_features(df):
    """Leak-free lags grouped by (season, player_id)."""
    df = df.sort_values(["season", "player_id", "gw"]).reset_index(drop=True)
    gb = df.groupby(["season", "player_id"], sort=False)
    
    # Points Lags
    df["form_1"] = gb["y"].shift(1)
    df["form_3"] = gb["y"].transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    df["form_5"] = gb["y"].transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
    df["ppg_lag"] = gb["y"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    
    # Stat Lags (Levels as of previous GW)
    lag_cols = ["xg", "xa", "def_actions_sum", "minutes", "starts", "goals", "assists", "bonus", "clean_sheets"]
    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = gb[col].shift(1)
            
    # Price/Market Lags
    if "price_m" in df.columns:
        df["price_lag1"] = gb["price_m"].shift(1)
    if "sel_count" in df.columns:
        df["sel_lag1"] = gb["sel_count"].shift(1)
        
    return df

def main():
    log("=" * 70)
    log("FPL TRAINING SET BUILDER -- FINAL (gws/ stacker)")
    log("=" * 70)
    
    seasons = discover_seasons()
    frames = []
    
    for sd in seasons:
        f = process_season(sd)
        if f is not None and len(f) > 0:
            frames.append(f)
            
    if not frames:
        raise RuntimeError("No data processed. Check logs above.")
        
    df = pd.concat(frames, ignore_index=True)
    log("-" * 70)
    log(f"Total raw rows: {len(df):,}")
    
    # Add leakage-free features
    df = add_lagged_features(df)
    
    # Drop rows where label is NaN (shouldn't happen after initial cleanup, but safety)
    df = df[df["y"].notna()].copy()
    
    # Save
    df.to_csv(OUT_CSV, index=False)
    
    # Summary
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"File saved: {OUT_CSV}")
    log(f"Rows: {len(df):,}")
    log(f"Seasons: {sorted(df['season'].unique())}")
    log(f"Players: {df['player_id'].nunique():,}")
    log(f"Features present: {[c for c in df.columns if '_lag' in c or c in ['xg','xa','fdr']]}")
    log("DONE.")

if __name__ == "__main__":
    main()