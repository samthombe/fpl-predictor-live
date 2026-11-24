# app.py -- FPL PREDICTOR: PROJECTIONS + SQUAD + CAPTAINCY + NEWS + CHIP PLANNER
import html
import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_clean_data
from predictor import project_players, LAST_DEBUG
from optimizer import optimize_squad, pick_starting_xi
from news_parser import availability_report
from chip_planner import get_all_fixtures, build_matrix, windows, swing, recommendations

st.set_page_config(page_title="FPL Edge", page_icon="🏆", layout="wide")

CSS = """
<style>
:root{
  --pitch:#00E5A0; --volt:#7CFF4F; --amber:#FFB020; --danger:#FF4D6D;
  --info:#38BDF8; --magenta:#F472B6; --ink:#0B1220; --surface:#121A2B; --line:#22304A;
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(0,229,160,.14), transparent 60%),
    radial-gradient(900px 420px at -10% 20%, rgba(56,189,248,.12), transparent 60%),
    linear-gradient(180deg,#0B1220 0%,#0D1526 55%,#0B1220 100%);
}
[data-testid="stHeader"]{background:transparent;}
.hero{
  padding:22px 26px;border:1px solid var(--line);border-radius:18px;margin-bottom:14px;
  background:linear-gradient(120deg,rgba(0,229,160,.16),rgba(56,189,248,.10) 45%,rgba(244,114,182,.12));
  box-shadow:0 10px 40px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.06);
}
.hero h1{margin:0;font-size:2rem;letter-spacing:.5px;
  background:linear-gradient(90deg,#7CFF4F,#00E5A0 40%,#38BDF8);
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{margin:6px 0 0;color:#B9C6DA;}
.gw-badge{display:inline-block;margin-top:10px;padding:6px 14px;border-radius:999px;
  border:1px solid rgba(0,229,160,.5);color:#7CFF4F;font-weight:700;letter-spacing:1px;
  background:rgba(0,229,160,.08);}
[data-testid="stTabs"] [role="tab"]{
  border-radius:12px 12px 0 0;border:1px solid var(--line);border-bottom:none;
  background:#0F1728;color:#9FB0C8;font-weight:600;padding:10px 16px;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{
  background:linear-gradient(180deg,rgba(0,229,160,.18),rgba(18,26,43,.9));
  color:#7CFF4F;border-color:rgba(0,229,160,.45);}
[data-testid="stMetric"]{
  background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:12px 16px;}
[data-testid="stMetricValue"]{color:#E8EEF7;}
[data-testid="stMetricLabel"]{color:#8FA2BC;}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.72rem;
  font-weight:800;letter-spacing:.6px;margin-right:8px;}
.pill-out{background:rgba(255,77,109,.16);color:#FF8FA3;border:1px solid rgba(255,77,109,.5);}
.pill-doubt{background:rgba(255,176,32,.16);color:#FFD08A;border:1px solid rgba(255,176,32,.5);}
.pill-monitor{background:rgba(56,189,248,.16);color:#9BDcF8;border:1px solid rgba(56,189,248,.5);}
.news-card{border:1px solid var(--line);border-left:4px solid var(--danger);
  background:var(--surface);border-radius:12px;padding:12px 14px;margin-bottom:10px;}
.news-card.doubt{border-left-color:var(--amber);}
.news-card .who{font-weight:800;color:#E8EEF7;}
.news-card .meta{color:#8FA2BC;font-size:.8rem;margin-left:6px;}
.news-card .txt{display:block;margin-top:6px;color:#C6D3E6;font-size:.88rem;line-height:1.45;}
.chip-card{border:1px solid var(--line);border-radius:14px;padding:14px 16px;height:100%;
  background:linear-gradient(160deg,rgba(18,26,43,.95),rgba(15,23,40,.9));}
.chip-card h4{margin:0 0 6px;color:#9FB0C8;font-size:.8rem;letter-spacing:1.2px;text-transform:uppercase;}
.chip-card .val{font-size:1.35rem;font-weight:800;color:#7CFF4F;}
.chip-card .why{color:#B9C6DA;font-size:.85rem;margin-top:6px;line-height:1.45;}
.chip-card .tg{color:#8FA2BC;font-size:.78rem;margin-top:8px;}
*::-webkit-scrollbar{width:10px;height:10px;}
*::-webkit-scrollbar-track{background:#0D1526;border-radius:8px;}
*::-webkit-scrollbar-thumb{background:linear-gradient(180deg,#00E5A0,#38BDF8);border-radius:8px;
  border:2px solid #0D1526;}
*::-webkit-scrollbar-thumb:hover{background:linear-gradient(180deg,#7CFF4F,#7DD3FC);}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner="Fetching FPL data...")
def get_data():
    return load_clean_data()


@st.cache_data(ttl=900, show_spinner="Loading fixture calendar...")
def get_fixtures_cached():
    return get_all_fixtures()


try:
    players, teams, events, fixtures, debug = get_data()
except Exception as e:
    st.error(f"Could not fetch FPL data: {e}")
    st.stop()

df = project_players(players, teams, fixtures)
next_gw = debug.get("next_gw")

st.markdown(
    f"<div class='hero'><h1>🏆 FPL EDGE</h1>"
    f"<p>Live ML projections · squad optimiser · captaincy matrix · injury awareness · chip timing.</p>"
    f"<span class='gw-badge'>GAMEWEEK {next_gw if next_gw else '—'}</span></div>",
    unsafe_allow_html=True,
)

with st.expander("🧪 Engine & fixture debug"):
    st.json(LAST_DEBUG)
with st.expander("🔎 API debug"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw elements", debug.get("raw_player_count", 0))
    c2.metric("Next GW", next_gw)
    c3.metric("Fixtures used", debug.get("fixtures_used", 0))
    c4.metric("Has now_cost", debug.get("has_now_cost"))
    st.json(debug.get("status_value_counts"))

if df.empty:
    st.error("No players after cleaning. Open the API debug panel.")
    st.stop()

tabs = st.tabs(["🔥 Projections", "🧠 Squad Builder", "🎯 Captaincy",
                "🚑 Availability & News", "🃏 Chip Planner", "⚖️ Transfer Lab"])
# ---------------- PROJECTIONS ----------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Available players", len(df))
    c2.metric("Fixtures loaded", len(fixtures) if fixtures else 0)
    c3.metric("Top projected", f"{df.iloc[0]['web_name']} ({df.iloc[0]['projected_points']:.2f})")
    st.subheader("Top 50 projected players")
    show_cols = ["web_name", "position", "team", "price_m", "projected_points",
                 "predicted_floor", "predicted_ceiling", "value", "form",
                 "points_per_game", "fixture_factor", "chance_pct", "status",
                 "selected_by_percent"]
    cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[cols].head(50).round(2), use_container_width=True, hide_index=True)
    st.download_button("Download projections CSV", df.to_csv(index=False).encode(),
                       "fpl_projections.csv", "text/csv")

# ---------------- SQUAD BUILDER ----------------
with tabs[1]:
    st.subheader("Generate optimal squad")
    budget = st.slider("Budget (£m)", 80.0, 100.0, 100.0, 0.5)
    max_team = st.selectbox("Max players per team", [2, 3], index=1)
    if st.button("Optimize 15-player team", type="primary"):
        import traceback
        try:
            d = df.sort_values("projected_points", ascending=False).reset_index(drop=True)
            with st.spinner("Building legal 15-man squad..."):
                squad = optimize_squad(d, budget=budget, max_per_team=max_team)
            xi = pick_starting_xi(squad)
            c1, c2, c3 = st.columns(3)
            c1.metric("Squad pts", f"{squad['projected_points'].sum():.2f}")
            c2.metric("Cost", f"£{squad['price_m'].sum():.1f}m")
            c3.metric("Bank", f"£{budget - squad['price_m'].sum():.1f}m")
            st.markdown(f"### ⚽ Starting XI — {xi['projected_points'].sum():.2f} pts")
            cols = [c for c in ["web_name", "position", "team", "price_m", "projected_points",
                                "predicted_floor", "predicted_ceiling", "selected_by_percent"]
                    if c in xi.columns]
            st.dataframe(xi[cols].round(2), use_container_width=True, hide_index=True)
            st.markdown("### 🪑 Full 15-man squad")
            st.dataframe(squad[cols].round(2), use_container_width=True, hide_index=True)
            st.download_button("Download squad CSV", squad.to_csv(index=False).encode(),
                               f"fpl_squad_gw{next_gw}.csv", "text/csv")
        except Exception as e:
            msg = str(e).strip() or f"{type(e).__name__} (no message)"
            st.error(f"Optimizer failed: {msg}")
            with st.expander("🔬 Traceback"):
                st.code(traceback.format_exc())

# ---------------- CAPTAINCY ----------------
with tabs[2]:
    st.subheader("Captaincy recommendations")
    cap = df.copy()
    if {"predicted_floor", "predicted_ceiling"} <= set(cap.columns):
        a, b, c, d = st.columns(4)
        with a:
            st.markdown("**🛡️ Safest (max floor)**")
            t = cap.nlargest(3, "predicted_floor")[["web_name", "position", "team", "predicted_floor"]]
            st.dataframe(t.round(2), hide_index=True, use_container_width=True)
        with b:
            st.markdown("**🚀 Upside (max ceiling)**")
            t = cap.nlargest(3, "predicted_ceiling")[["web_name", "position", "team", "predicted_ceiling"]]
            st.dataframe(t.round(2), hide_index=True, use_container_width=True)
        with c:
            st.markdown("**💎 Differentials (<10% own)**")
            if "selected_by_percent" in cap.columns:
                t = cap[(cap["selected_by_percent"] < 10) & (cap["predicted_ceiling"] > 7)] \
                    .nlargest(3, "predicted_ceiling")
                if not t.empty:
                    t = t[["web_name", "position", "team", "predicted_ceiling", "selected_by_percent"]]
                    t.columns = ["Player", "Pos", "Team", "Ceiling", "Own%"]
                    st.dataframe(t.round(2), hide_index=True, use_container_width=True)
                else:
                    st.write("No strong differentials this week.")
            else:
                st.warning("Ownership data missing.")
        with d:
            st.markdown("**⚖️ Value (ceiling / £m)**")
            v = cap.head(50).copy()
            if "price_m" in v.columns and v["price_m"].min() > 0:
                v["cpm"] = v["predicted_ceiling"] / v["price_m"]
                t = v.nlargest(3, "cpm")[["web_name", "team", "predicted_ceiling", "price_m", "cpm"]]
                t.columns = ["Player", "Team", "Ceiling", "Price", "Pts/£m"]
                st.dataframe(t.round(2), hide_index=True, use_container_width=True)
    else:
        st.warning("Captaincy matrix needs floor/ceiling from the ML models.")

# ---------------- AVAILABILITY & NEWS ----------------
with tabs[3]:
    st.subheader("Availability & injury news")
    flagged, summ = availability_report(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚫 Out", summ["out"])
    c2.metric("⚠️ Doubt", summ["doubt"])
    c3.metric("👁 Monitor", summ["monitor"])
    c4.metric("🆕 Fresh (24h)", summ["fresh24"])
    st.caption("Projections are already chance-adjusted; this panel shows WHY and how recent the news is.")

    f1, f2 = st.columns([1, 2])
    tier = f1.selectbox("Risk tier", ["All", "Out", "Doubt", "Monitor"])
    q = f2.text_input("Search player / club / news", "")
    view = flagged.copy()
    if tier != "All":
        view = view[view["risk_tier"] == tier]
    if q:
        ql = q.lower()
        mask = (view["web_name"].str.lower().str.contains(ql)
                | view["team"].astype(str).str.lower().str.contains(ql)
                | view["news"].str.lower().str.contains(ql))
        view = view[mask]

    cards = view[view["risk_tier"].isin(["Out", "Doubt"])].head(6)
    for _, r in cards.iterrows():
        cls = "news-card" + (" doubt" if r["risk_tier"] == "Doubt" else "")
        pill = "pill-doubt" if r["risk_tier"] == "Doubt" else "pill-out"
        age = r["news_age_h"]
        age_txt = f"{age:.0f}h ago" if age is not None else "date n/a"
        st.markdown(
            f"<div class='{cls}'><span class='pill {pill}'>{r['risk_tier'].upper()}</span>"
            f"<span class='who'>{html.escape(str(r['web_name']))}</span>"
            f"<span class='meta'>{html.escape(str(r['team']))} · {r['position']} · "
            f"{r['chance_pct']:.0f}% · {age_txt}</span>"
            f"<span class='txt'>{html.escape(str(r['news']) or 'No detail provided.')}</span></div>",
            unsafe_allow_html=True,
        )

    if view.empty:
        st.success("No flagged players match this filter — squad availability is clean.")
    else:
        cols = [c for c in ["web_name", "team", "position", "status_label", "risk_tier",
                            "chance_pct", "news_age_h", "projected_points", "news"]
                if c in view.columns]
        st.dataframe(view[cols].round(1), use_container_width=True, hide_index=True)

# ---------------- CHIP PLANNER ----------------
with tabs[4]:
    st.subheader("Chip timing & fixture outlook")
    horizon = st.slider("Look-ahead (gameweeks)", 3, 8, 6)
    start = int(next_gw or 1)
    try:
        all_fx = get_fixtures_cached()
    except Exception as e:
        st.error(f"Could not load fixture calendar: {e}")
        all_fx = []

    if all_fx:
        disp, fdr, cnt = build_matrix(all_fx, teams, start, horizon)
        rec, blanks, doubles, playing = recommendations(fdr, cnt, df, start, horizon)

        st.markdown("#### 🃏 Recommended chip windows")
        cols = st.columns(4)
        for col, chip in zip(cols, ["Bench Boost", "Triple Captain", "Free Hit", "Wildcard"]):
            r = rec[chip]
            targets = ", ".join(r["targets"]) or "—"
            col.markdown(
                f"<div class='chip-card'><h4>{chip}</h4>"
                f"<div class='val'>{html.escape(str(r['headline']))}</div>"
                f"<div class='why'>{html.escape(str(r['rationale']))}</div>"
                f"<div class='tg'>Focus: {html.escape(targets)}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### 📅 Blank / Double gameweeks")
        st.dataframe(windows(cnt), use_container_width=True, hide_index=True)

        st.markdown("#### 🗓 Fixture difficulty matrix (next %d GWs)" % horizon)
        st.dataframe(disp, use_container_width=True)
        def _fdr_css(v):
            if pd.isna(v):
                return "background-color:#121A2B;color:#5A6B85;"
            stops = {1: (0, 229, 160), 2: (124, 255, 79), 3: (255, 176, 32),
                     4: (255, 122, 69), 5: (255, 77, 109)}
            k = int(max(1, min(5, round(float(v)))))
            r, g, b = stops[k]
            return f"background-color:rgba({r},{g},{b},0.22);color:#E8EEF7;font-weight:700;"

        styled = fdr.style.map(_fdr_css).format(lambda v: "—" if pd.isna(v) else f"{v:.1f}")
        st.dataframe(styled, use_container_width=True)

        st.markdown("#### 🔄 Fixture swings (easing = negative delta)")
        sw = swing(fdr, start, horizon).sort_values("delta")
        st.dataframe(sw.head(12).round(2), use_container_width=True, hide_index=True)
    else:
        st.info("Fixture calendar unavailable; chip windows cannot be computed right now.")

st.caption("Data: official FPL API · Models retrain automatically every Sunday 02:00 UTC.")

# ---------------- TRANSFER LAB ----------------
from transfer_simulator import analyze_transfer, CAPTAIN_NONE, CAPTAIN_IN, CAPTAIN_SWAP

with tabs[5]:
    st.subheader("Transfer Lab — hit analysis")
    st.caption("Sell one player, buy another, and see exactly when the points hit pays for itself.")

    names = sorted(df["web_name"].unique().tolist())
    c1, c2, c3 = st.columns(3)
    with c1:
        out_name = st.selectbox("Player OUT (sell)", names, index=0)
    with c2:
        in_name = st.selectbox("Player IN (buy)", names, index=1 if len(names) > 1 else 0)
    with c3:
        hit = st.selectbox("Points hit", [0, 4, 8], index=1,
                           format_func=lambda h: "Free (0)" if h == 0 else f"-{h}")

    c4, c5 = st.columns(2)
    with c4:
        horizon = st.slider("Horizon (gameweeks)", 2, 8, 4)
    with c5:
        cap_label = st.selectbox("Captain scenario",
                                 ["No captain", "IN is captain", "Captain swap (both x2)"])
    cap_code = {"No captain": CAPTAIN_NONE,
                "IN is captain": CAPTAIN_IN,
                "Captain swap (both x2)": CAPTAIN_SWAP}[cap_label]

    if st.button("Analyze transfer", type="primary"):
        if out_name == in_name:
            st.error("Pick two different players.")
        else:
            start = int(next_gw or 1)
            try:
                all_fx = get_fixtures_cached()
                _, fdr, cnt = build_matrix(all_fx, teams, start, horizon)
                res = analyze_transfer(df, fdr, cnt, out_name, in_name,
                                       int(hit), cap_code, start, horizon)
            except Exception as e:
                st.error(f"Analysis failed: {type(e).__name__}: {e}")
                res = None

            if res is None:
                st.warning("Could not locate one of the players in the fixture matrix.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric(f"GW{start} swing", f"{res['tbl']['gain'].iloc[0]:+.2f}")
                m2.metric("Break-even", f"GW{res['breakeven']}" if res["breakeven"] else "Not in horizon")
                m3.metric(f"Net by GW{int(res['tbl']['GW'].iloc[-1])}", f"{res['net_end']:+.2f}")

                banner = {"good": st.success, "marginal": st.warning,
                          "bad": st.error, "free": st.info}[res["kind"]]
                banner(res["verdict"])

                chart = res["tbl"].set_index("GW")[["cum", "net"]].rename(
                    columns={"cum": "Cumulative gain", "net": "Net after hit"})
                st.line_chart(chart)
                st.dataframe(res["tbl"].round(2), use_container_width=True, hide_index=True)

                st.caption(
                    f"GW{start} uses the live model projection. GW{start + 1}+ are extrapolated by "
                    "scaling each player's neutral rate by their club's fixture difficulty and "
                    "double/blank gameweeks. Form changes and injuries after this week are not modelled."
                )