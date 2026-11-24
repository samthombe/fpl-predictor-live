# build_training_set.py -- ALIGNED WITH LIVE API FEATURES
# Counting stats = season-to-date totals as of PRIOR GW (matches bootstrap-static).
# Price in £m, ownership in %, position as int. Leak-free.
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

BASE_DIR = Path(__file__).resolve().parent
HISTORY_DIR = BASE_DIR / "data" / "vaastav" / "data"
OUT_DIR = BASE_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "training_set.csv"
SEASONS_TO_USE = 3

CUM_COLS = ["minutes", "starts", "goals", "assists", "bonus",
            "clean_sheets", "xg", "xa", "def_actions_sum"]


def log(msg):
    print(msg, flush=True)


def discover_seasons():
    if not HISTORY_DIR.exists():
        raise FileNotFoundError(f"History dir not found: {HISTORY_DIR}")
    pat = re.compile(r"20\d\d-\d\d")
    dirs = [p for p in HISTORY_DIR.iterdir() if p.is_dir() and pat.search(p.name)]
    dirs.sort(key=lambda d: d.name, reverse=True)
    log(f"Discovered {len(dirs)} season(s): {[d.name for d in dirs]}")
    return dirs[:SEASONS_TO_USE]


def load_gw_files(season_dir):
    gws_dir = season_dir / "gws"
    if not gws_dir.exists():
        log(f"  [{season_dir.name}] No 'gws/' folder.")
        return None
    csvs = sorted(gws_dir.glob("gw*.csv"))
    frames = []
    for f in csvs:
        try:
            df = pd.read_csv(f, low_memory=False)
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
    raw = load_gw_files(season_dir)
    if raw is None or raw.empty:
        return None

    out = pd.DataFrame()
    out["player_id"] = pd.to_numeric(raw.get("element"), errors="coerce")
    out["gw"] = pd.to_numeric(raw.get("round"), errors="coerce")
    out["y"] = pd.to_numeric(raw.get("total_points"), errors="coerce")
    out["season"] = name
    out = out.dropna(subset=["player_id", "gw", "y"])
    out["player_id"] = out["player_id"].astype(int)
    out["gw"] = out["gw"].astype(int)

    # Position: try gw file, then cleaned_players.csv
    pos = None
    for c in ["element_type", "position"]:
        if c in raw.columns:
            pos = pd.to_numeric(raw[c], errors="coerce")
            break
    if pos is None:
        cp = season_dir / "cleaned_players.csv"
        if cp.exists():
            cpd = pd.read_csv(cp)
            if {"id", "element_type"} <= set(cpd.columns):
                mapper = dict(zip(pd.to_numeric(cpd["id"], errors="coerce"),
                                  pd.to_numeric(cpd["element_type"], errors="coerce")))
                pos = out["player_id"].map(mapper)
    out["position"] = pos if pos is not None else np.nan

    # Per-GW counting stats
    src = {
        "minutes": "minutes", "starts": "starts", "goals": "goals_scored",
        "assists": "assists", "bonus": "bonus", "clean_sheets": "clean_sheets",
        "xg": "expected_goals", "xa": "expected_assists",
    }
    for dst, s in src.items():
        out[dst] = pd.to_numeric(raw.get(s), errors="coerce").fillna(0)

    def_cols = [c for c in ["tackles", "interceptions", "blocks", "clearances"] if c in raw.columns]
    if def_cols:
        out["def_actions_sum"] = raw[def_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    elif "defensive_contribution" in raw.columns:
        out["def_actions_sum"] = pd.to_numeric(raw["defensive_contribution"], errors="coerce").fillna(0)
    else:
        out["def_actions_sum"] = 0.0

    # Price: vaastav 'value' is tenths of £m -> convert to £m (matches live)
    val = pd.to_numeric(raw.get("value"), errors="coerce")
    out["price_m"] = val / 10.0 if val.median() > 25 else val

    # Ownership: 'selected' is owner COUNT -> approximate % of managers
    sel = pd.to_numeric(raw.get("selected"), errors="coerce").fillna(0)
    gw_total = sel.groupby(out["gw"]).transform("sum") / 15.0  # ~15 picks per manager
    out["sel_pct"] = (sel / gw_total.replace(0, np.nan) * 100.0).clip(0, 100).fillna(0)

    # Context
    out["is_home"] = pd.to_numeric(raw.get("was_home"), errors="coerce").fillna(0).astype(int)

    # FDR from fixtures.csv
    out["fdr"] = 3.0
    fix_path = season_dir / "fixtures.csv"
    if fix_path.exists():
        try:
            fx = pd.read_csv(fix_path)
            gw_c = next((c for c in fx.columns if c.lower() in ["round", "event"]), None)
            h_c = next((c for c in fx.columns if c.lower() in ["team_h_code", "home_team"]), None)
            a_c = next((c for c in fx.columns if c.lower() in ["team_a_code", "away_team"]), None)
            hd_c = next((c for c in fx.columns if "diff" in c.lower() and "h" in c.lower()), None)
            ad_c = next((c for c in fx.columns if "diff" in c.lower() and "a" in c.lower()), None)
            if gw_c and h_c and a_c:
                lookup = {}
                for _, r in fx.iterrows():
                    g = int(r[gw_c])
                    hd = float(r[hd_c]) if hd_c and not pd.isna(r[hd_c]) else 3.0
                    ad = float(r[ad_c]) if ad_c and not pd.isna(r[ad_c]) else 3.0
                    lookup[(g, int(r[h_c]))] = hd
                    lookup[(g, int(r[a_c]))] = ad
                # player's team unknown in gw files; use opponent-agnostic mean of the two? 
                # Safer: leave 3.0 unless we can map team. Keep default.
        except Exception as e:
            log(f"  [{name}] FDR calc failed: {e}")

    out = out.drop_duplicates(subset=["season", "player_id", "gw"])
    log(f"  [{name}] Processed {len(out)} rows.")
    return out


def add_aligned_features(df):
    """Season-to-date totals as of PRIOR GW + form lags. Matches live API semantics."""
    df = df.sort_values(["season", "player_id", "gw"]).reset_index(drop=True)
    gb = df.groupby(["season", "player_id"], sort=False)

    df["form_1"] = gb["y"].shift(1)
    df["form_3"] = gb["y"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["form_5"] = gb["y"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["ppg_lag"] = gb["y"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

    # CUMULATIVE through prior GW (this is what bootstrap-static sends live)
    for col in CUM_COLS:
        df[f"{col}_lag1"] = gb[col].transform(
            lambda s: s.shift(1).expanding(min_periods=1).sum()
        ).fillna(0)

    df["price_lag1"] = gb["price_m"].shift(1)
    df["sel_lag1"] = gb["sel_pct"].shift(1)
    return df


def main():
    log("=" * 70)
    log("FPL TRAINING SET BUILDER -- LIVE-ALIGNED FEATURES")
    log("=" * 70)
    frames = [f for sd in discover_seasons() if (f := process_season(sd)) is not None]
    if not frames:
        raise RuntimeError("No data processed.")
    df = pd.concat(frames, ignore_index=True)
    df = add_aligned_features(df)
    df = df[df["y"].notna()].copy()
    df.to_csv(OUT_CSV, index=False)

    log("=" * 70)
    log(f"Rows: {len(df):,} | Players: {df['player_id'].nunique():,}")
    log("Feature ranges (should now match live API scale):")
    log(df[["minutes_lag1", "goals_lag1", "price_lag1", "sel_lag1"]]
        .describe().round(2).to_string())
    log("DONE.")


if __name__ == "__main__":
    main()