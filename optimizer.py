import pandas as pd
import numpy as np

# FPL squad composition is FIXED at 15 players:
# 2 GK, 5 DEF, 5 MID, 3 FWD. (The 3-5 ranges only apply to the starting XI.)
SQUAD_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["id", "position", "price_m", "projected_points", "team_code"])
    df = df[df["position"].isin(SQUAD_COUNTS.keys())]
    df = df[(df["projected_points"] > 0) & (df["price_m"] > 0)]
    df = df.drop_duplicates(subset=["id"])
    df = df.reset_index(drop=True)
    if len(df) < 15:
        raise ValueError("Not enough available players to build a 15-man squad.")
    return df


def _arrays(df: pd.DataFrame):
    pos = df["position"].to_numpy()
    team = df["team_code"].astype(int).to_numpy()
    price = df["price_m"].astype(float).to_numpy()
    pts = df["projected_points"].astype(float).to_numpy()
    value = np.where(price > 0, pts / price, 0.0)
    groups = {p: np.where(pos == p)[0].tolist() for p in SQUAD_COUNTS}
    return pos, team, price, pts, value, groups


def _team_cap_ok(team_used, tc, ts, max_per_team):
    # after removing s (team ts) and adding c (team tc), team[tc] must stay <= max
    used_before_add = team_used[tc] - (1 if tc == ts else 0)
    return used_before_add < max_per_team


def _construct(order, groups, team, price, pts, value, max_per_team, budget):
    """Greedy build by 'pts' or 'value', then repair budget."""
    key_arr = pts if order == "pts" else value
    team_used = {}
    chosen = {p: [] for p in SQUAD_COUNTS}
    cost = 0.0

    for p in ["GK", "DEF", "MID", "FWD"]:
        need = SQUAD_COUNTS[p]
        cands = sorted(groups[p], key=lambda i: key_arr[i], reverse=True)
        for i in cands:
            t = int(team[i])
            if team_used.get(t, 0) < max_per_team:
                chosen[p].append(i)
                team_used[t] = team_used.get(t, 0) + 1
                cost += price[i]
                if len(chosen[p]) == need:
                    break
        if len(chosen[p]) < need:
            return None  # not enough feasible players in this position

    # Repair budget: swap expensive -> cheaper same-position, min points loss
    guard = 0
    while cost > budget and guard < 500:
        guard += 1
        best = None  # (loss, pos, idx_in_list, replacement, new_cost)
        for p in chosen:
            cur = chosen[p]
            cur_set = set(cur)
            for si, s in enumerate(cur):
                ts = int(team[s])
                for c in groups[p]:
                    if c in cur_set or c == s:
                        continue
                    if price[c] >= price[s]:
                        continue
                    new_cost = cost - price[s] + price[c]
                    if new_cost > budget:
                        continue
                    tc = int(team[c])
                    if not _team_cap_ok(team_used, tc, ts, max_per_team):
                        continue
                    loss = pts[s] - pts[c]
                    if best is None or loss < best[0]:
                        best = (loss, p, si, c, new_cost, ts, tc)
        if best is None:
            break
        _, p, si, c, new_cost, ts, tc = best
        chosen[p][si] = c
        team_used[ts] -= 1
        team_used[tc] = team_used.get(tc, 0) + 1
        cost = new_cost

    if cost > budget:
        return None
    flat = chosen["GK"] + chosen["DEF"] + chosen["MID"] + chosen["FWD"]
    return flat, team_used, cost


def _local_search(flat, team, price, pts, groups, max_per_team, budget):
    chosen = {p: [] for p in SQUAD_COUNTS}
    for i in flat:
        chosen[str(team.dtype and "") or ""]  # noop placeholder
    # rebuild grouped
    chosen = {p: [] for p in SQUAD_COUNTS}
    pos_of = {}
    # we need position per index; recompute from groups membership
    idx_to_pos = {}
    for p, lst in groups.items():
        for i in lst:
            idx_to_pos[i] = p
    for i in flat:
        chosen[idx_to_pos[i]].append(i)

    team_used = {}
    for p in chosen:
        for i in chosen[p]:
            t = int(team[i])
            team_used[t] = team_used.get(t, 0) + 1
    cost = sum(price[i] for i in flat)

    improved = True
    rounds = 0
    while improved and rounds < 200:
        improved = False
        rounds += 1
        best_gain = 0.0
        best_move = None
        for p in chosen:
            cur = chosen[p]
            cur_set = set(cur)
            for si, s in enumerate(cur):
                ts = int(team[s])
                for c in groups[p]:
                    if c in cur_set or c == s:
                        continue
                    gain = pts[c] - pts[s]
                    if gain <= best_gain:
                        continue
                    tc = int(team[c])
                    if not _team_cap_ok(team_used, tc, ts, max_per_team):
                        continue
                    new_cost = cost - price[s] + price[c]
                    if new_cost > budget:
                        continue
                    best_gain = gain
                    best_move = (p, si, c, ts, tc, new_cost)
        if best_move is not None:
            p, si, c, ts, tc, new_cost = best_move
            chosen[p][si] = c
            team_used[ts] -= 1
            team_used[tc] = team_used.get(tc, 0) + 1
            cost = new_cost
            improved = True

    flat = chosen["GK"] + chosen["DEF"] + chosen["MID"] + chosen["FWD"]
    return flat, sum(pts[i] for i in flat)


def optimize_squad(df: pd.DataFrame, budget: float = 100.0, max_per_team: int = 3) -> pd.DataFrame:
    """
    Dependency-free FPL squad optimizer.
    Builds a valid 15-man squad (2 GK / 5 DEF / 5 MID / 3 FWD),
    respects budget and max-per-team, and hill-climbs to maximise projected points.
    """
    df = _prepare(df)
    pos, team, price, pts, value, groups = _arrays(df)

    best_flat = None
    best_total = -1.0

    for order in ("pts", "value"):
        built = _construct(order, groups, team, price, pts, value, max_per_team, budget)
        if built is None:
            continue
        flat, _, _ = built
        flat, total = _local_search(flat, team, price, pts, groups, max_per_team, budget)
        if total > best_total:
            best_total = total
            best_flat = flat

    if best_flat is None or len(best_flat) != 15:
        raise RuntimeError(
            "Optimizer could not build a valid 15-player squad. "
            "Increase the budget, set max players per team to 3, "
            "or ensure enough players are available per position."
        )

    squad = df.iloc[best_flat].copy()
    squad["_pos_order"] = squad["position"].map(POS_ORDER)
    squad = (
        squad.sort_values(["_pos_order", "projected_points"], ascending=[True, False])
        .drop(columns=["_pos_order"])
        .reset_index(drop=True)
    )
    return squad