import math
import pandas as pd


def calc_tank_wn8(player_stats: dict, expected: dict) -> dict | None:
    """
    Calculate WN8 for a single tank.

    player_stats: {battles, wins, damage_dealt, frags, spotted, dropped_capture_points}
    expected:     {expDmg, expFrag, expSpot, expDef, expWinRate}

    Returns dict with wn8, raw ratios, clamped ratios, and % of expected per metric.
    Returns None if battles == 0 or any expected value is 0.
    """
    battles = player_stats.get("battles", 0)
    if not battles or battles == 0:
        return None

    # Guard against zero expected values
    if any(expected.get(k, 0) <= 0 for k in ("expDmg", "expFrag", "expSpot", "expWinRate")):
        return None

    dmg_pb   = player_stats.get("damage_dealt", 0) / battles
    frag_pb  = player_stats.get("frags", 0) / battles
    spot_pb  = player_stats.get("spotted", 0) / battles
    def_pb   = player_stats.get("dropped_capture_points", 0) / battles
    wr       = player_stats.get("wins", 0) / battles  # decimal

    exp_dmg  = expected["expDmg"]
    exp_frag = expected["expFrag"]
    exp_spot = expected["expSpot"]
    exp_def  = expected["expDef"] if expected["expDef"] > 0 else 0.001
    exp_wr   = expected["expWinRate"]

    rDMG  = dmg_pb  / exp_dmg
    rFRAG = frag_pb / exp_frag
    rSPOT = spot_pb / exp_spot
    rDEF  = def_pb  / exp_def
    rWIN  = wr      / exp_wr

    # Clamped/adjusted ratios
    rDMG_c  = max(0.0, (rDMG  - 0.22) / (1 - 0.22))
    rFRAG_c = max(0.0, min(rDMG_c + 0.2, (rFRAG - 0.12) / (1 - 0.12)))
    rSPOT_c = max(0.0, min(rDMG_c + 0.1, (rSPOT - 0.38) / (1 - 0.38)))
    rDEF_c  = max(0.0, min(rDMG_c + 0.1, (rDEF  - 0.10) / (1 - 0.10)))
    rWIN_c  = max(0.0, (rWIN  - 0.71) / (1 - 0.71))

    wn8 = (
        980  * rDMG_c
        + 210  * rDMG_c  * rFRAG_c
        + 155  * rFRAG_c * rSPOT_c
        + 75   * rDEF_c  * rFRAG_c
        + 145  * min(1.8, rWIN_c)
    )

    return {
        "wn8":            wn8,
        "rDMG":           rDMG,
        "rFRAG":          rFRAG,
        "rSPOT":          rSPOT,
        "rDEF":           rDEF,
        "rWIN":           rWIN,
        "rDMG_c":         rDMG_c,
        "rFRAG_c":        rFRAG_c,
        "rSPOT_c":        rSPOT_c,
        "rDEF_c":         rDEF_c,
        "rWIN_c":         rWIN_c,
        "dmg_per_battle": dmg_pb,
        "frags_per_battle": frag_pb,
        "spots_per_battle": spot_pb,
        "def_per_battle":   def_pb,
        "winrate":          wr,
        "dmg_pct_of_exp":  rDMG  * 100,
        "frag_pct_of_exp": rFRAG * 100,
        "spot_pct_of_exp": rSPOT * 100,
        "def_pct_of_exp":  rDEF  * 100,
        "win_pct_of_exp":  rWIN  * 100,
    }


def build_tank_dataframe(
    tank_stats: list,
    expected_values: dict,
    tankopedia: dict,
) -> pd.DataFrame:
    """
    Build a DataFrame joining API tank stats with expected values and tankopedia.
    Tanks missing from expected_values get wn8=NaN.
    Tanks with 0 battles are skipped.
    """
    rows = []
    for entry in tank_stats:
        tank_id = entry.get("tank_id")
        if tank_id is None:
            continue
        a = entry.get("all", {})
        battles = a.get("battles", 0)
        if not battles:
            continue

        meta = tankopedia.get(tank_id, {})
        row = {
            "tank_id":    tank_id,
            "name":       meta.get("name", f"Unknown #{tank_id}"),
            "tier":       meta.get("tier", 0),
            "nation":     meta.get("nation", "?"),
            "type":       meta.get("type", "?"),
            "is_premium": meta.get("is_premium", False),
            "battles":           battles,
            "wins":              a.get("wins", 0),
            "damage_dealt":      a.get("damage_dealt", 0),
            "frags":             a.get("frags", 0),
            "survived_battles":  a.get("survived_battles", 0),
            "spotted":           a.get("spotted", 0),
            "dropped_capture_points": a.get("dropped_capture_points", 0),
        }

        exp = expected_values.get(tank_id)
        if exp:
            row["exp_dmg"] = exp.get("expDmg", float("nan"))
            calc = calc_tank_wn8(a, exp)
            if calc:
                row.update(calc)
            else:
                row["wn8"] = float("nan")
        else:
            row["exp_dmg"] = float("nan")
            row["wn8"] = float("nan")

        # Fill missing calc fields with NaN for tanks without expected values
        for col in ("dmg_per_battle", "frags_per_battle", "spots_per_battle",
                    "def_per_battle", "winrate",
                    "rDMG_c", "rFRAG_c", "rSPOT_c", "rDEF_c", "rWIN_c",
                    "dmg_pct_of_exp", "frag_pct_of_exp", "spot_pct_of_exp",
                    "def_pct_of_exp", "win_pct_of_exp"):
            if col not in row:
                row[col] = float("nan")

        row["wn8_impact"] = (
            row["wn8"] * battles
            if not (isinstance(row["wn8"], float) and math.isnan(row["wn8"]))
            else 0.0
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def calc_overall_wn8(df: pd.DataFrame) -> float:
    """Weighted average WN8 by battles (only tanks with valid WN8)."""
    valid = df.dropna(subset=["wn8"])
    if valid.empty:
        return 0.0
    total_weighted = (valid["wn8"] * valid["battles"]).sum()
    total_battles = valid["battles"].sum()
    return total_weighted / total_battles if total_battles > 0 else 0.0


def calc_drag_score(df: pd.DataFrame, overall_wn8: float) -> pd.DataFrame:
    """
    Add drag_score = (overall_wn8 - tank_wn8) * (battles / total_battles).
    Positive means the tank is pulling overall WN8 down.
    Returns only tanks with valid WN8, sorted by drag_score descending.
    """
    valid = df.dropna(subset=["wn8"]).copy()
    valid["drag_score"] = (overall_wn8 - valid["wn8"]) * valid["battles"]
    return valid.sort_values("drag_score", ascending=False)


def calc_goal_wn8(
    overall_wn8_now: float,
    total_battles_now: int,
    future_battles: int,
    goal: float = 2900.0,
) -> dict:
    """
    Calculate what WN8 you need on your next `future_battles` battles to reach `goal`.

    Formula:
        required = goal + (goal - current_wn8) * current_battles / future_battles

    Returns:
        required_future_wn8: float
        achievable: bool (True if required <= 5000)
        battles_at_4000: int — battles needed at WN8=4000 to reach goal
    """
    if future_battles <= 0:
        return {"required_future_wn8": float("inf"), "achievable": False, "battles_at_4000": None}

    if overall_wn8_now >= goal:
        return {"required_future_wn8": 0.0, "achievable": True, "battles_at_4000": 0}

    required = goal + (goal - overall_wn8_now) * total_battles_now / future_battles
    achievable = required <= 5000

    # How many battles at WN8=4000 to reach goal?
    wn8_at_4k = 4000.0
    if wn8_at_4k > goal:
        # n * 4000 + current_battles * current_wn8 >= goal * (n + current_battles)
        # n * (4000 - goal) >= current_battles * (goal - current_wn8)
        # n >= current_battles * (goal - current_wn8) / (4000 - goal)
        battles_at_4000 = math.ceil(
            total_battles_now * (goal - overall_wn8_now) / (wn8_at_4k - goal)
        )
    else:
        battles_at_4000 = None

    return {
        "required_future_wn8": required,
        "achievable": achievable,
        "battles_at_4000": battles_at_4000,
    }


def calc_tank_target_stats(expected: dict, goal_wn8: float = 2900.0) -> dict:
    """
    Compute what per-battle stats a player needs to hit goal_wn8 on a specific tank.

    Assumes equal improvement across all metrics (same rX_c value = x),
    with rWIN_c capped at 1.8. Standard WN8 then collapses to

        WN8 = 980*x + (210 + 155 + 75)*x**2 + 145*min(1.8, x)

    solved for x as a quadratic on whichever side of the win cap applies.
    """
    def solve(lin: float, const: float) -> float:
        # 440*x^2 + lin*x + const - goal = 0, positive root
        return (-lin + math.sqrt(lin * lin + 4 * 440.0 * (goal_wn8 - const))) / (2 * 440.0)

    x = solve(980.0 + 145.0, 0.0)
    if x > 1.8:
        x = solve(980.0, 145.0 * 1.8)

    win_c = min(1.8, x)

    return {
        "target_dmg":   expected["expDmg"]     * (0.22 + x    * 0.78),
        "target_frags": expected["expFrag"]    * (0.12 + x    * 0.88),
        "target_spots": expected["expSpot"]    * (0.38 + x    * 0.62),
        "target_def":   expected["expDef"]     * (0.10 + x    * 0.90),
        "target_wr":    expected["expWinRate"] * (0.71 + win_c * 0.29),  # decimal
    }


def calc_tank_target_dmg_actual(exp_dmg: float, rFRAG: float, rSPOT: float,
                                rDEF: float, rWIN: float, goal_wn8: float = 2900.0) -> float:
    """
    Damage per battle needed to hit goal_wn8 on a tank, holding the player's
    actual frag / spot / defense / win ratios on that tank fixed.

    WN8 rises monotonically with the damage ratio (the frag, spot and def terms
    are capped at rDMG_c + 0.2 / 0.1, so they can only grow with it), which
    makes bisection on rDMG_c safe.
    """
    f_raw = (rFRAG - 0.12) / 0.88
    s_raw = (rSPOT - 0.38) / 0.62
    e_raw = (rDEF - 0.10) / 0.90
    w = min(1.8, max(0.0, (rWIN - 0.71) / 0.29))

    def wn8_at(d: float) -> float:
        f = max(0.0, min(d + 0.2, f_raw))
        sp = max(0.0, min(d + 0.1, s_raw))
        e = max(0.0, min(d + 0.1, e_raw))
        return 980 * d + 210 * d * f + 155 * f * sp + 75 * e * f + 145 * w

    lo, hi = 0.0, 10.0
    if wn8_at(lo) >= goal_wn8:
        return exp_dmg * 0.22
    for _ in range(60):
        mid = (lo + hi) / 2
        if wn8_at(mid) < goal_wn8:
            lo = mid
        else:
            hi = mid
    return exp_dmg * (0.22 + hi * 0.78)


def calc_metric_gaps(df: pd.DataFrame, overall_wn8: float, goal: float = 2900.0) -> dict:
    """
    For each of the 5 WN8 metrics, compute the weighted-average current clamped ratio
    and what that ratio would need to be to push overall WN8 to `goal`
    (holding all other metrics constant at their current weighted averages).

    WN8 = 980*D + 210*D*F + 155*F*S + 75*E*F + 145*min(1.8, W)

    Returns dict keyed by metric with {current, needed, label, pct_change}
    """
    valid = df.dropna(subset=["wn8", "rDMG_c"]).copy()
    if valid.empty:
        return {}

    total_b = valid["battles"].sum()

    def wavg(col):
        return (valid[col] * valid["battles"]).sum() / total_b

    cur = {
        "D": wavg("rDMG_c"),
        "F": wavg("rFRAG_c"),
        "S": wavg("rSPOT_c"),
        "E": wavg("rDEF_c"),
        "W": wavg("rWIN_c"),
    }

    def wn8_of(c):
        return (980 * c["D"] + 210 * c["D"] * c["F"] + 155 * c["F"] * c["S"]
                + 75 * c["E"] * c["F"] + 145 * min(1.8, c["W"]))

    labels = {
        "D": "Damage",
        "F": "Frags (Kills)",
        "S": "Spots",
        "E": "Defense",
        "W": "Win Rate",
    }

    results = {}
    for metric in labels:
        # WN8 is linear in any single metric with the others fixed:
        # wn8 = base + slope * metric  →  solve for the metric that hits goal.
        base = wn8_of({**cur, metric: 0.0})
        slope = wn8_of({**cur, metric: 1.0}) - base
        needed_raw = (goal - base) / slope if slope > 0 else float("inf")
        # For win rate, cap display at 1.8 (the formula cap)
        if metric == "W":
            needed_raw = min(needed_raw, 1.8)

        current_val = cur[metric]
        pct_change = ((needed_raw - current_val) / current_val * 100) if current_val > 0 else float("inf")

        results[metric] = {
            "label":      labels[metric],
            "current":    current_val,
            "needed":     needed_raw,
            "pct_change": pct_change,
        }

    return results
