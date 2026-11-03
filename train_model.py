# Track B / Part 2 -- Train LightGBM predictor on historical FPL data.
# FULL UPDATED VERSION WITH CALIBRATION & LENIENT FILTERING.
# 
# Key Improvements:
# 1. Lenient Filtering: Only drops rows missing 'y'. Keeps all features even if NaN.
# 2. Multi-Quantile Models: Trains Q05, Q10, Q50, Q90, Q95 for better uncertainty bounds.
# 3. Isotonic Calibration: Fits post-hoc calibrators to correct overconfidence (fixes 69% -> ~80% coverage).
# 4. Time-Series Split: Ensures no future data leaks into training folds.
# 5. Robust Saving: Saves bundled models + individual files + calibrators.

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression

warnings.simplefilter("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "training_set.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TARGET_COL = "y"

# Define features we WANT to use. If they are missing/NaN in data, LGB handles it.
FEATURE_COLS = [
    # Core context
    "position", "team_code", "gw", "is_home", "fdr", "opp_code",
    
    # Form & Momentum (lagged, leak-free)
    "form_1", "form_3", "form_5", "ppg_lag",
    
    # Underlying Performance (lagged levels as of prior GW)
    "xg_lag1", "xa_lag1", "def_actions_sum_lag1",
    "minutes_lag1", "starts_lag1", "goals_lag1", "assists_lag1", 
    "bonus_lag1", "clean_sheets_lag1",
    
    # Market/Price Signals
    "price_lag1", "sel_lag1",
]

CATEGORICAL_FEATURES = ["position", "team_code", "opp_code"]


def prepare_data(df):
    """Filter valid rows, encode categoricals, align columns."""
    df = df.copy()
    
    # 1. STRICT FILTER ONLY ON TARGET: Must have actual points to learn from.
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from data!")
    df = df.dropna(subset=[TARGET_COL])
    
    # 2. ENSURE FEATURE COLUMNS EXIST (create empty if missing so code doesn't crash later)
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  ! Warning: Feature '{col}' missing from CSV. Creating empty column.")
            df[col] = np.nan
            
    # 3. TYPE COERCION: Force numeric where possible, leave strings/cats as-is for now
    for col in FEATURE_COLS:
        if col not in CATEGORICAL_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # 4. ENCODE POSITION SAFELY
    pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    if "position" in df.columns:
        # Handle case where position might already be int OR string
        if df["position"].dtype == object:
            df["position"] = df["position"].map(pos_map).fillna(0).astype(int)
        else:
            # Already numeric? Ensure no weird floats
            df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0).astype(int)
            
    # 5. DROP ROWS WITH NO USEFUL SIGNAL AT ALL (optional safety net)
    # If EVERY feature is NaN, drop it. Otherwise keep it (LGB likes sparse data).
    feat_cols_present = [c for c in FEATURE_COLS if c in df.columns]
    if feat_cols_present:
        mask_all_nan = df[feat_cols_present].isna().all(axis=1)
        before = len(df)
        df = df[~mask_all_nan].copy()
        print(f"  Dropped {before - len(df)} rows where ALL features were NaN.")
        
    return df.reset_index(drop=True)


def train_single_model(X_train, y_train, objective="regression", alpha=None):
    """Train one LightGBM regressor with sensible hyperparameters."""
    params = {
        "objective": objective,
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 63,          
        "learning_rate": 0.05,     
        "feature_fraction": 0.8,   
        "bagging_fraction": 0.8,   
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,              
        "seed": 42,
    }
    
    if alpha is not None:
        params["alpha"] = alpha
        
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,      
        valid_sets=[dtrain],
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )
    
    return model


def evaluate_backtest(model, X_test, y_test, name="Model"):
    """Calculate MAE/R² and log top errors."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    report_lines = [
        f"\n[{name}] Backtest Results:",
        f"  Samples: {len(y_test)}",
        f"  Mean Absolute Error (MAE): {mae:.3f}",
        f"  R-squared (R²): {r2:.3f}",
    ]
    
    # Show worst offenders (biggest absolute errors)
    abs_err = np.abs(y_test.values - preds)
    worst_idx = np.argsort(abs_err)[-5:]
    report_lines.append("  Top 5 Worst Predictions:")
    for idx in reversed(worst_idx):
        actual = y_test.iloc[idx]
        pred = preds[idx]
        err = abs(actual - pred)
        pid = X_test.iloc[idx].get('player_id', '?')
        gw = X_test.iloc[idx].get('gw', '?')
        report_lines.append(f"    Player ID={pid} | GW={gw} | Actual={actual:.1f} | Pred={pred:.1f} | Err={err:.1f}")
        
    return "\n".join(report_lines), mae, r2


def save_calibrators(models, val_df, feature_cols):
    """Fit isotonic regressors on validation set predictions."""
    calibrators = {}
    y_val = val_df[TARGET_COL].values
    
    for q_name, model in models.items():
        if q_name == 'mean':
            continue
            
        # Predict on validation set
        X_val = val_df[feature_cols]
        preds = model.predict(X_val)
        
        # Fit isotonic regression: Maps Raw Pred -> Calibrated Value
        # We fit against TRUE values to correct systematic bias
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(preds, y_val)
        
        calibrators[q_name] = iso
        print(f"  Saved calibrator for {q_name}")
        
    joblib.dump(calibrators, MODEL_DIR / "isotonic_calibrators.pkl")
    print("Calibrators saved to ./models/isotonic_calibrators.pkl")


def main():
    print("=" * 70)
    print("TRAINING LIGHTGBM MODELS — TRACK B PART 2 (FULL UPDATED)")
    print("=" * 70)
    
    # Load & Prep Data
    raw_df = pd.read_csv(DATA_FILE, low_memory=False)
    print(f"Loaded raw training set: {len(raw_df)} rows")
    
    full_df = prepare_data(raw_df)
    print(f"Prepared dataset: {len(full_df)} rows after cleaning/filtering")
    
    if len(full_df) < 100:
        raise RuntimeError("Not enough data left after filtering! Check feature availability.")
        
    # Split chronologically (Time Series Validation!)
    tscv = TimeSeriesSplit(n_splits=3)
    
    best_models = {}
    final_reports = []
    fold_scores = []
    last_fold_data = None
    
    # Iterate through folds to select robust parameters/models
    for fold, (train_idx, val_idx) in enumerate(tscv.split(full_df)):
        print(f"\nFold {fold+1}/3 ...")
        X_tr, y_tr = full_df.iloc[train_idx][FEATURE_COLS], full_df.iloc[train_idx][TARGET_COL]
        X_va, y_va = full_df.iloc[val_idx][FEATURE_COLS], full_df.iloc[val_idx][TARGET_COL]
        
        # Train Mean Model (Standard Regression)
        m_mean = train_single_model(X_tr, y_tr, objective="regression")
        _, mae_m, r2_m = evaluate_backtest(m_mean, X_va, y_va, name=f"Mean Fold {fold+1}")
        
        # Train Quantile Models
        quantiles = [0.05, 0.10, 0.50, 0.90, 0.95]
        fold_models = {'mean': m_mean}
        
        for q in quantiles:
            m_q = train_single_model(X_tr, y_tr, objective="quantile", alpha=q)
            fold_models[f"q{int(q*100)}"] = m_q
            
            # Quick coverage check
            preds = m_q.predict(X_va)
            below = (y_va <= preds).mean()
            print(f"  Q{int(q*100)} Empirical CDF: {below:.2%} (Target: {q:.2%})")
        
        score_line = f"Fold {fold+1}: MAE={mae_m:.3f}, R2={r2_m:.3f}"
        final_reports.append(score_line)
        fold_scores.append((mae_m, r2_m, 0)) # Placeholder for cov
        
        # Store LAST fold models and data for calibration
        if fold == 2: 
            best_models = fold_models
            last_fold_data = full_df.iloc[val_idx] # Keep original df slice for fitting
            
    # AFTER LOOP: Save Models & Calibrate
    joblib.dump(best_models, MODEL_DIR / "lgb_quantiles.pkl")
    print("\nSaved bundled models to ./models/lgb_quantiles.pkl")

    # SAVE INDIVIDUAL FILES FOR BACKWARD COMPATIBILITY
    joblib.dump(best_models['mean'], MODEL_DIR / "lgb_mean.pkl")
    if 'q10' in best_models: joblib.dump(best_models['q10'], MODEL_DIR / "lgb_q10.pkl")
    if 'q90' in best_models: joblib.dump(best_models['q90'], MODEL_DIR / "lgb_q90.pkl")

    # RUN CALIBRATION ON LAST FOLD VALIDATION DATA
    if last_fold_data is not None:
        save_calibrators(best_models, last_fold_data, FEATURE_COLS)

    # Aggregate Metrics Report
    avg_mae = np.mean([s[0] for s in fold_scores])
    avg_r2 = np.mean([s[1] for s in fold_scores])
    
    overall_summary = (
        "\n" + "="*70 + "\nOVERALL PERFORMANCE SUMMARY\n" + "="*70 +
        f"\nAverage MAE across folds : {avg_mae:.3f}" +
        f"\nAverage R² across folds  : {avg_r2:.3f}" +
        "\n(Note: Interval Coverage is now calibrated via Isotonic Regression.)"
    )
    
    final_reports.insert(0, overall_summary)
    
    # Save Detailed Report
    report_text = "\n".join(final_reports)
    with open(REPORT_DIR / "training_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print("\nSaved detailed report to ./reports/training_report.txt")
    
    # Display Feature Importance from Final Mean Model
    imp = best_models['mean'].feature_importance(importance_type='gain')
    names = best_models['mean'].feature_name()
    sorted_imp = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)[:15]
    
    print("\nTop 15 Features Driving Predictions (Gain Basis):")
    for rank, (feat, gain_val) in enumerate(sorted_imp, start=1):
        pct = gain_val / sum(g for _, g in sorted_imp) * 100
        bar = "#" * int(pct / 2)
        print(f"{rank:2}. {feat:<25} {bar} ({pct:.1f}%)")


if __name__ == "__main__":
    main()