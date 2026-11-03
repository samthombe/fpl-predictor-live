import streamlit as st
import pandas as pd

from data_loader import load_clean_data
from predictor import project_players
from optimizer import optimize_squad


st.set_page_config(page_title="FPL Predictor MVP", page_icon="🏆", layout="wide")
st.title("🏆 FPL Predictor MVP")
st.caption("Live projections + optimal 15-player squad for the next Gameweek.")


@st.cache_data(ttl=900, show_spinner="Fetching FPL data...")
def get_data():
    return load_clean_data()


try:
    players, teams, events, fixtures, debug = get_data()
except Exception as e:
    st.error(f"Could not fetch FPL data: {e}")
    st.stop()

with st.expander("🔎 Debug: what the FPL API returned"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw elements", debug.get("raw_player_count", 0))
    c2.metric("Next GW", debug.get("next_gw"))
    c3.metric("Fixtures used", debug.get("fixtures_used", 0))
    c4.metric("Has now_cost", debug.get("has_now_cost"))
    st.markdown("**Top-level keys:**")
    st.code(str(debug.get("top_level_keys")))
    st.markdown("**Status value counts (raw):**")
    st.json(debug.get("status_value_counts"))
    if debug.get("sample_element"):
        st.markdown("**Sample player record:**")
        st.json(debug.get("sample_element"))

df = project_players(players, teams, fixtures)

if "_relaxed" in df.columns and len(df) > 0 and bool(df["_relaxed"].any()):
    st.warning("Strict availability filter removed everyone; showing all priced players instead.")

if df.empty:
    st.error("Still no players after cleaning. Open the Debug panel and paste the sample record.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Available players", len(df))
col2.metric("Fixtures loaded", len(fixtures) if fixtures else 0)
col3.metric("Top projected", f"{df.iloc[0]['web_name']} ({df.iloc[0]['projected_points']:.2f})")

st.subheader("🔥 Top 50 projected players")
show_cols = [
    "web_name", "position", "team", "price_m", 
    "projected_points", "predicted_floor", "predicted_ceiling", # NEW ML COLUMNS
    "value", "form", "points_per_game", "fixture_factor", "chance_pct", "status",
]
# Filter columns that actually exist in the dataframe to avoid KeyError
available_show_cols = [c for c in show_cols if c in df.columns]
st.dataframe(df[available_show_cols].head(50).round(2), use_container_width=True)

st.subheader("🧠 Generate optimal squad")
budget = st.slider("Budget (£m)", 80.0, 100.0, 100.0, 0.5)
max_team = st.selectbox("Max players per team", [2, 3], index=1)

if st.button("Optimize 15-player team"):
    try:
        with st.spinner("Solving optimization problem..."):
            squad = optimize_squad(df, budget=budget, max_per_team=max_team)
        
        total_pts = squad["projected_points"].sum()
        total_cost = squad["price_m"].sum()
        
        # Check if floor/ceiling columns exist before summing
        has_floor = "predicted_floor" in squad.columns
        has_ceiling = "predicted_ceiling" in squad.columns
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Projected points", f"{total_pts:.2f}")
        c2.metric("Squad cost", f"£{total_cost:.1f}m")
        c3.metric("Bank", f"£{budget - total_cost:.1f}m")
        
        if has_floor and has_ceiling:
            c4, c5 = st.columns(2)
            c4.metric("Floor (Safe)", f"{squad['predicted_floor'].sum():.2f}")
            c5.metric("Ceiling (Upside)", f"{squad['predicted_ceiling'].sum():.2f}")

        display_cols = ["web_name", "position", "team", "price_m", "projected_points"]
        if has_floor:
            display_cols.append("predicted_floor")
        if has_ceiling:
            display_cols.append("predicted_ceiling")
        display_cols.extend(["value", "chance_pct"])
        
        final_display_cols = [c for c in display_cols if c in squad.columns]
        
        st.dataframe(squad[final_display_cols].round(2), use_container_width=True)
        
        st.download_button(
            "Download squad CSV",
            data=squad.to_csv(index=False).encode(),
            file_name="fpl_optimal_squad.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(str(e))

st.info("MVP uses heuristic projections. Next milestone: ML model on historical data.")
# --- CAPTAINCY MATRIX ---
st.subheader("🎯 Captaincy Recommendations")

cap_df = df.copy()
if "predicted_floor" in cap_df.columns and "predicted_ceiling" in cap_df.columns:
    cap_df["gap"] = cap_df["predicted_ceiling"] - cap_df["predicted_floor"]
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("**🛡️ Safest Picks (Max Floor)**")
        safest = cap_df.nlargest(3, "predicted_floor")[["web_name", "position", "team", "predicted_floor"]]
        st.dataframe(safest.round(2), hide_index=True, use_container_width=True)
        
    with col_b:
        st.markdown("**🚀 High Upside (Max Ceiling)**")
        upside = cap_df.nlargest(3, "predicted_ceiling")[["web_name", "position", "team", "predicted_ceiling"]]
        st.dataframe(upside.round(2), hide_index=True, use_container_width=True)
        
    # In app.py, replace the Differential logic with this:

with col_c:
    st.markdown("**💎 True Differentials (High Ceiling, Low Ownership)**")
    # Filter for players owned by < 10% but with ceiling > 7.0
    diff_candidates = cap_df[
        (cap_df["sel_lag1"] < 10.0) & 
        (cap_df["predicted_ceiling"] > 7.0)
    ].nlargest(3, "predicted_ceiling")
    
    if not diff_candidates.empty:
        diffs = diff_candidates[["web_name", "position", "team", "predicted_ceiling", "sel_lag1"]]
        st.dataframe(diffs.round(2), hide_index=True, use_container_width=True)
    else:
        st.write("No strong differentials found (all high-ceiling players are popular).")
else:
    st.warning("Captaincy matrix requires ML models to be loaded.")
    # In app.py sidebar
st.sidebar.header("⚙️ Manual Overrides")
override_injuries = st.sidebar.multiselect(
    "Mark Players as OUT/Doubtful",
    options=df['web_name'].unique(),
    help="Use this if scraper misses recent news."
)

# Apply overrides in predictor
if override_injuries:
    df.loc[df['web_name'].isin(override_injuries), 'chance_pct'] = 0
    df.loc[df['web_name'].isin(override_injuries), 'status'] = 'i' # injured
