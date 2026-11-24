# chip_planner.py -- FIXED matrix orientation + chip timing
import requests
import numpy as np
import pandas as pd

FIXTURE_URL = "https://fantasy.premierleague.com/api/fixtures/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://fantasy.premierleague.com/",
}


def get_all_fixtures():
    r = requests.get(FIXTURE_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _maps(teams):
    id_to_code, code_to_short = {}, {}
    if teams is not None and not teams.empty:
        if {"id", "code"} <= set(teams.columns):
            t = teams[["id", "code"]].apply(pd.to_numeric, errors="coerce").dropna()
            id_to_code = {int(r.id): int(r.code) for r in t.itertuples(index=False)}
        name_col = "short_name" if "short_name" in teams.columns else ("name" if "name" in teams.columns else None)
        if name_col and "code" in teams.columns:
            t = teams[["code", name_col]].copy()
            t["code"] = pd.to_numeric(t["code"], errors="coerce")
            t = t.dropna(subset=["code"])
            code_to_short = {int(r.code): str(getattr(r, name_col)) for r in t.itertuples(index=False)}
    return id_to_code, code_to_short


def _int_cols(df):
    d = df.copy()
    try:
        d.columns = [int(c) for c in d.columns]
    except (TypeError, ValueError):
        pass
    return d


def build_matrix(fixtures, teams, start_gw, horizon=6):
    start_gw, horizon = int(start_gw), int(horizon)
    gws = list(range(start_gw, start_gw + horizon))
    id2code, code2short = _maps(teams)
    team_names = sorted(code2short.values())
    cells = {t: {g: [] for g in gws} for t in team_names}

    for f in fixtures or []:
        try:
            ev = int(f.get("event"))
        except (TypeError, ValueError):
            continue
        if ev not in gws:
            continue
        try:
            h = id2code.get(int(f.get("team_h")))
            a = id2code.get(int(f.get("team_a")))
        except (TypeError, ValueError):
            continue
        hs, as_ = code2short.get(h), code2short.get(a)
        hd = float(f.get("team_h_difficulty") or 3)
        ad = float(f.get("team_a_difficulty") or 3)
        if hs:
            cells[hs][ev].append((as_ or "?", hd, "H"))
        if as_:
            cells[as_][ev].append((hs or "?", ad, "A"))

    disp, fdr, cnt = {}, {}, {}
    for t in team_names:
        disp[t], fdr[t], cnt[t] = {}, {}, {}
        for g in gws:
            e = cells[t][g]
            cnt[t][g] = len(e)
            if not e:
                disp[t][g], fdr[t][g] = "—", np.nan
            elif len(e) == 1:
                opp, d, ha = e[0]
                disp[t][g], fdr[t][g] = f"{opp}({ha})", d
            else:
                disp[t][g] = f"{len(e)}× " + "/".join(f"{o}({ha})" for o, _, ha in e)
                fdr[t][g] = float(np.mean([d for _, d, _ in e]))

    # dict-of-dicts -> columns=teams, index=gws; transpose -> rows=teams, cols=gws
    disp_df = pd.DataFrame(disp).T.sort_index(axis=1)
    fdr_df = pd.DataFrame(fdr).T.sort_index(axis=1)
    cnt_df = pd.DataFrame(cnt).T.sort_index(axis=1)
    for d in (disp_df, fdr_df, cnt_df):
        d.index.name = "team"
        d.columns = [int(c) for c in d.columns]
    return disp_df, fdr_df, cnt_df


def windows(count_df):
    count_df = _int_cols(count_df)
    rows = []
    for g in count_df.columns:
        blanks = [t for t in count_df.index if count_df.loc[t, g] == 0]
        doubles = [t for t in count_df.index if count_df.loc[t, g] >= 2]
        rows.append({"GW": int(g), "Playing": int((count_df[g] > 0).sum()),
                     "Blanks": len(blanks), "Blank teams": ", ".join(blanks) or "—",
                     "Doubles": len(doubles), "Double teams": ", ".join(doubles) or "—"})
    return pd.DataFrame(rows)


def swing(fdr_df, start_gw, horizon):
    fdr_df = _int_cols(fdr_df)
    start_gw = int(start_gw)
    gws = [int(g) for g in fdr_df.columns]
    first = [g for g in gws if g < start_gw + 3]
    later = [g for g in gws if g >= start_gw + 3]
    a = fdr_df[first].mean(axis=1) if first else pd.Series(np.nan, index=fdr_df.index)
    b = fdr_df[later].mean(axis=1) if later else pd.Series(np.nan, index=fdr_df.index)
    out = pd.DataFrame({"first3": a, "next3": b})
    out["delta"] = out["next3"] - out["first3"]
    return out.reset_index().rename(columns={"index": "team"})


def recommendations(fdr_df, count_df, projections, start_gw, horizon):
    fdr_df, count_df = _int_cols(fdr_df), _int_cols(count_df)
    start_gw = int(start_gw)
    gws = [int(g) for g in fdr_df.columns]
    blanks = {g: [t for t in count_df.index if count_df.loc[t, g] == 0] for g in gws}
    doubles = {g: [t for t in count_df.index if count_df.loc[t, g] >= 2] for g in gws}
    playing = {g: int((count_df[g] > 0).sum()) for g in gws}
    rec = {}

    if gws:
        bb = max(gws, key=lambda g: 3 * len(doubles[g]) + playing[g])
        rec["Bench Boost"] = {"gw": bb, "headline": f"GW{bb}",
                              "rationale": f"{len(doubles[bb])} teams double; {playing[bb]}/20 clubs play.",
                              "targets": doubles[bb][:6]}
        fh = max(gws, key=lambda g: (len(blanks[g]), len(doubles[g])))
        kind = "Blank" if blanks[fh] else "Double"
        rec["Free Hit"] = {"gw": fh, "headline": f"GW{fh}",
                           "rationale": f"{kind} gameweek: {len(blanks[fh])} blank, {len(doubles[fh])} double.",
                           "targets": (blanks[fh] or doubles[fh])[:6]}
    else:
        rec["Bench Boost"] = {"gw": None, "headline": "—", "rationale": "No window", "targets": []}
        rec["Free Hit"] = {"gw": None, "headline": "—", "rationale": "No window", "targets": []}

    tc_best = None
    if projections is not None and not projections.empty and "predicted_ceiling" in projections.columns:
        for _, r in projections.nlargest(8, "predicted_ceiling").iterrows():
            team = r.get("team")
            if team not in fdr_df.index:
                continue
            for g in gws:
                c = count_df.loc[team, g]
                if c == 0:
                    continue
                d = fdr_df.loc[team, g]
                ease = 1.25 - 0.10 * (3.0 if pd.isna(d) else float(d))
                score = float(r["predicted_ceiling"]) * (1.8 if c >= 2 else 1.0) * ease
                if tc_best is None or score > tc_best[0]:
                    tc_best = (score, r["web_name"], team, g, c)
    if tc_best:
        _, name, team, g, c = tc_best
        rec["Triple Captain"] = {"gw": g, "headline": f"{name} · GW{g}",
                                 "rationale": "Doubles this GW." if c >= 2 else "Easiest fixture in the window.",
                                 "targets": [team]}
    else:
        rec["Triple Captain"] = {"gw": None, "headline": "—", "rationale": "No candidate", "targets": []}

    sw = swing(fdr_df, start_gw, horizon)
    improving = sw[sw["delta"] < 0].sort_values("delta")
    wc_g = start_gw + 3 if len(improving) >= 6 else start_gw + 1
    rec["Wildcard"] = {"gw": wc_g, "headline": f"Before GW{wc_g}",
                       "rationale": f"{len(improving)} clubs' fixtures ease over the window.",
                       "targets": improving["team"].head(6).tolist()}
    return rec, blanks, doubles, playing