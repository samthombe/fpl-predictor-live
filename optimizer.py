# optimizer.py -- ROBUST SQUAD BUILDER (never fails silently)
import pandas as pd
import numpy as np

REQUIRED = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE = 15


class SquadError(Exception):
    """Raised only when the input data itself is unusable."""
    pass


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Hard requirements
    for col in ["web_name", "position", "team", "price_m", "projected_points"]:
        if col not in d.columns:
            raise SquadError(f"Missing required column: '{col}'")

    d["price_m"] = pd.to_numeric(d["price_m"], errors="coerce")
    d["projected_points"] = pd.to_numeric(d["projected_points"], errors="coerce")

    d = d.dropna(subset=["price_m", "projected_points"])
    d = d[(d["price_m"] > 0) & (d["position"].isin(REQUIRED.keys()))]

    # Normalise team labels so the 3-per-club rule works even if names differ
    d["team"] = d["team"].astype(str).fillna("?")

    # De-duplicate players (same name+team+price can appear twice after merges)
    d = d.drop_duplicates(subset=["web_name", "team", "position"], keep="first")

    return d.reset_index(drop=True)


def _counts(squad):
    pos = {k: 0 for k in REQUIRED}
    teams = {}
    for p in squad:
        pos[p["position"]] += 1
        teams[p["team"]] = teams.get(p["team"], 0) + 1
    return pos, teams


def _can_add(player, squad, budget_left, max_per_team):
    pos, teams = _counts(squad)
    if len(squad) >= SQUAD_SIZE:
        return False
    if pos[player["position"]] >= REQUIRED[player["position"]]:
        return False
    if teams.get(player["team"], 0) >= max_per_team:
        return False
    if player["price_m"] > budget_left + 1e-9:
        return False
    return True


def _cheapest_filler(d, position, squad, budget_left, max_per_team):
    """Find cheapest legal player of a given position (used to guarantee feasibility)."""
    pos, teams = _counts(squad)
    used = {p["web_name"] + "|" + p["team"] for p in squad}
    pool = d[d["position"] == position].sort_values("price_m")
    for _, r in pool.iterrows():
        key = r["web_name"] + "|" + r["team"]
        if key in used:
            continue
        if teams.get(r["team"], 0) >= max_per_team:
            continue
        if r["price_m"] <= budget_left + 1e-9:
            return r.to_dict()
    return None


def optimize_squad(df: pd.DataFrame, budget: float = 100.0, max_per_team: int = 3) -> pd.DataFrame:
    d = _prep(df)

    if d.empty:
        raise SquadError("No usable players after cleaning (check price_m / position columns).")

    # Feasibility check: do we even have enough players per position?
    for pos, need in REQUIRED.items():
        have = int((d["position"] == pos).sum())
        if have < need:
            raise SquadError(
                f"Not enough {pos} players available: need {need}, found {have}. "
                f"Loosen the availability filter in data_loader.py."
            )

    # ---- PHASE 1: reserve budget with cheapest legal skeleton ----
    squad = []
    budget_left = float(budget)

    for pos, need in REQUIRED.items():
        for _ in range(need):
            pick = _cheapest_filler(d, pos, squad, budget_left, max_per_team)
            if pick is None:
                raise SquadError(
                    f"Cannot build a legal squad within £{budget:.1f}m "
                    f"(max {max_per_team}/club). Try raising the budget or allowing 3 per team."
                )
            squad.append(pick)
            budget_left -= pick["price_m"]

    # ---- PHASE 2: upgrade each slot to best affordable points ----
    # Sort candidates by projected points descending once
    ranked = d.sort_values("projected_points", ascending=False).to_dict("records")

    improved = True
    passes = 0
    while improved and passes < 6:
        improved = False
        passes += 1
        for i in range(len(squad)):
            current = squad[i]
            rest = squad[:i] + squad[i + 1:]
            freed = budget_left + current["price_m"]

            pos_rest, teams_rest = _counts(rest)
            used = {p["web_name"] + "|" + p["team"] for p in rest}

            best = None
            for cand in ranked:
                if cand["position"] != current["position"]:
                    continue
                key = cand["web_name"] + "|" + cand["team"]
                if key in used:
                    continue
                if teams_rest.get(cand["team"], 0) >= max_per_team:
                    continue
                if cand["price_m"] > freed + 1e-9:
                    continue
                if cand["projected_points"] <= current["projected_points"] + 1e-9:
                    break  # ranked desc -> nothing better below
                best = cand
                break

            if best is not None:
                squad[i] = best
                budget_left = freed - best["price_m"]
                improved = True

    out = pd.DataFrame(squad)

    # Order nicely: GK, DEF, MID, FWD then by points
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    out["_o"] = out["position"].map(order)
    out = out.sort_values(["_o", "projected_points"], ascending=[True, False]).drop(columns="_o")

    # Sanity assertions with REAL messages
    if len(out) != SQUAD_SIZE:
        raise SquadError(f"Internal error: built {len(out)} players instead of 15.")
    if out["price_m"].sum() > budget + 1e-6:
        raise SquadError(f"Internal error: squad cost £{out['price_m'].sum():.1f}m exceeds £{budget:.1f}m.")

    return out.reset_index(drop=True)


def pick_starting_xi(squad: pd.DataFrame) -> pd.DataFrame:
    """Best legal XI: 1 GK, >=3 DEF, >=2 MID, >=1 FWD, total 11."""
    s = squad.copy()
    xi = []
    # mandatory minimums
    mins = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
    for pos, n in mins.items():
        pool = s[s["position"] == pos].sort_values("projected_points", ascending=False)
        xi.extend(pool.head(n).to_dict("records"))

    chosen = {p["web_name"] + "|" + p["team"] for p in xi}
    rest = [r for r in s.to_dict("records") if (r["web_name"] + "|" + r["team"]) not in chosen]
    rest.sort(key=lambda r: r["projected_points"], reverse=True)

    for r in rest:
        if len(xi) >= 11:
            break
        # cannot add a second GK
        if r["position"] == "GK":
            continue
        xi.append(r)

    out = pd.DataFrame(xi)
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    out["_o"] = out["position"].map(order)
    return out.sort_values(["_o", "projected_points"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)