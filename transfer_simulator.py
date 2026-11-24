# transfer_simulator.py -- MULTI-GW TRANSFER HIT ANALYSIS
import numpy as np
import pandas as pd

CAPTAIN_NONE, CAPTAIN_IN, CAPTAIN_SWAP = 0, 1, 2


def _fdr_mult(d):
    if d is None or pd.isna(d):
        return 1.0
    return float(np.clip(1.18 - 0.07 * (float(d) - 1), 0.85, 1.20))


def _row_for(df, name):
    m = df[df["web_name"] == name]
    return None if m.empty else m.iloc[0]


def _gw_mults(count_s, fdr_s, gws):
    rows = []
    for g in gws:
        n = int(count_s.get(g, 0) or 0)
        d = fdr_s.get(g, np.nan)
        blank = (n == 0)
        fm = 1.0 if blank else _fdr_mult(d)
        dm = 1.0 + 0.75 * max(n - 1, 0)
        rows.append((g, fm, dm, blank))
    return rows


def project_horizon(player, count_s, fdr_s, start, horizon):
    """GW+1 = live projection; later GWs = neutral rate x fixture x doubles."""
    gws = list(range(int(start), int(start) + int(horizon)))
    mults = _gw_mults(count_s, fdr_s, gws)
    p_next = float(player.get("projected_points", 0) or 0)
    fm0, dm0 = mults[0][1], mults[0][2]
    base = p_next / (fm0 * dm0) if (fm0 * dm0) > 0 else p_next
    rows = []
    for (g, fm, dm, blank) in mults:
        rows.append({"GW": g, "projected": 0.0 if blank else base * fm * dm})
    return pd.DataFrame(rows)


def analyze_transfer(df, fdr_df, count_df, out_name, in_name, hit,
                     captain_code, start, horizon):
    a, b = _row_for(df, out_name), _row_for(df, in_name)
    if a is None or b is None:
        return None

    def _series(team, frame, default):
        if team in frame.index:
            return frame.loc[team]
        return pd.Series(default, index=frame.columns)

    ha = project_horizon(a, _series(a["team"], count_df, 1), _series(a["team"], fdr_df, 3.0), start, horizon)
    hb = project_horizon(b, _series(b["team"], count_df, 1), _series(b["team"], fdr_df, 3.0), start, horizon)

    mA = mB = 1.0
    if captain_code == CAPTAIN_IN:
        mB = 2.0
    elif captain_code == CAPTAIN_SWAP:
        mA = mB = 2.0

    out_pts = ha["projected"].to_numpy() * mA
    in_pts = hb["projected"].to_numpy() * mB
    gain = in_pts - out_pts
    cum = np.cumsum(gain)
    net = cum - float(hit)

    breakeven = None
    if hit > 0:
        for i, c in enumerate(cum):
            if c >= hit - 1e-9:
                breakeven = int(ha["GW"].iloc[i])
                break

    tbl = pd.DataFrame({
        "GW": ha["GW"].to_numpy(),
        "OUT_pts": out_pts,
        "IN_pts": in_pts,
        "gain": gain,
        "cum": cum,
        "net": net,
    })

    net_end = float(net[-1])
    last_gw = int(tbl["GW"].iloc[-1])
    if hit == 0:
        kind = "free"
        verdict = ("Free transfer — pure upside." if net_end > 0
                   else "Free transfer, but the outlook is negative.")
    elif breakeven is not None and net_end > 0:
        kind = "good"
        verdict = (f"WORTH IT — break-even in GW{breakeven}; "
                   f"net {net_end:+.1f} pts by GW{last_gw}.")
    elif breakeven is not None:
        kind = "marginal"
        verdict = (f"MARGINAL — recovers the hit by GW{breakeven} but ends "
                   f"{net_end:+.1f} over the horizon.")
    else:
        kind = "bad"
        verdict = (f"NOT WORTH IT — never recovers the -{hit} within "
                   f"{horizon} GWs (net {net_end:+.1f}).")

    return {"tbl": tbl, "breakeven": breakeven, "net_end": net_end,
            "hit": int(hit), "kind": kind, "verdict": verdict,
            "out": out_name, "in": in_name}