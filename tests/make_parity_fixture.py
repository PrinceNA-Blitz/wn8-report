"""
Write tests/parity_cases.json: WN8 cases worked out by the Python code, which
tests/wn8_parity.mjs then runs through site_assets/wn8.js.

    python tests/make_parity_fixture.py           rewrite the file
    python tests/make_parity_fixture.py --check   exit 1 if the file is out of date

The cases are seeded, so the file only changes when the Python WN8 code or
this script changes. It holds:

- "cases": single tanks, hand-picked edge cases plus seeded random ones.
- "accounts": the random tanks as one account, scored against two goals (one
  above the overall WN8, one below), in the same form as a --dump-parity file.
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
# Nothing here needs Streamlit; blocking it skips its cache warnings on import.
sys.modules.setdefault("streamlit", None)

from wn8_calculator import build_tank_dataframe, calc_overall_wn8, calc_tank_wn8  # noqa: E402
from export_player_html import EV_FIELDS, STAT_FIELDS, parity_payload  # noqa: E402

OUT = os.path.join(ROOT, "tests", "parity_cases.json")
SEED = 20261107
RANDOM_CASES = 300

# A typical tier 8-10 tank: expDmg, expFrag, expSpot, expDef, expWinRate.
TYPICAL = [1500.0, 0.8, 1.1, 0.6, 0.52]

#          name                                  stats [b, wins, dmg, frags, spots, def, survived]   expected values
EDGE_CASES = [
    ("no battles",                             [0, 0, 0, 0, 0, 0, 0],                      TYPICAL),
    ("no expected values",                     [50, 30, 80000, 40, 60, 20, 25],            None),
    ("expDmg is zero",                         [50, 30, 80000, 40, 60, 20, 25],            [0.0, 0.8, 1.1, 0.6, 0.52]),
    ("expFrag is zero",                        [50, 30, 80000, 40, 60, 20, 25],            [1500.0, 0.0, 1.1, 0.6, 0.52]),
    ("expSpot is zero",                        [50, 30, 80000, 40, 60, 20, 25],            [1500.0, 0.8, 0.0, 0.6, 0.52]),
    ("expWinRate is zero",                     [50, 30, 80000, 40, 60, 20, 25],            [1500.0, 0.8, 1.1, 0.6, 0.0]),
    ("expDef is zero, uses 0.001",             [50, 30, 80000, 40, 60, 20, 25],            [1500.0, 0.8, 1.1, 0.0, 0.52]),
    ("expDef is zero, no defence points",      [50, 30, 80000, 40, 60, 0, 25],             [1500.0, 0.8, 1.1, 0.0, 0.52]),
    ("all stats zero except battles",          [10, 0, 0, 0, 0, 0, 0],                     TYPICAL),
    ("damage below the 0.22 floor",            [10, 5, 2000, 8, 11, 6, 5],                 TYPICAL),
    ("frag cap binds",                         [10, 5, 5000, 40, 10, 0, 3],                TYPICAL),
    ("spot cap binds",                         [10, 5, 9000, 8, 90, 3, 4],                 TYPICAL),
    ("defence cap binds",                      [10, 5, 9000, 8, 11, 60, 4],                TYPICAL),
    ("win rate above the 1.8 cap",             [20, 20, 40000, 20, 25, 10, 15],            [1500.0, 0.8, 1.1, 0.6, 0.45]),
    ("one battle",                             [1, 1, 2500, 2, 1, 0, 1],                   TYPICAL),
    ("large tank",                             [3000, 2000, 6000000, 3000, 3500, 1500, 1800], TYPICAL),
]


def python_wn8(stats, ev):
    """calc_tank_wn8 on the compact arrays; None when it returns None."""
    result = calc_tank_wn8(dict(zip(STAT_FIELDS, stats)),
                           dict(zip(EV_FIELDS, ev)) if ev is not None else {})
    return None if result is None else result["wn8"]


def random_tank(rng):
    """One tank with plausible expected values and stats spread around them."""
    exp_def = 0.0 if rng.random() < 0.1 else rng.uniform(0.05, 1.5)
    ev = [rng.uniform(300, 3500), rng.uniform(0.3, 1.5), rng.uniform(0.4, 2.0),
          exp_def, rng.uniform(0.42, 0.62)]
    battles = rng.choice([1, 2, 3, 5, 8]) if rng.random() < 0.25 else rng.randint(1, 3000)
    per_battle_def = exp_def if exp_def > 0 else 0.5
    stats = [
        battles,
        min(battles, round(ev[4] * rng.uniform(0.5, 1.9) * battles)),
        round(ev[0] * rng.uniform(0.0, 2.2) * battles),
        round(ev[1] * rng.uniform(0.0, 2.2) * battles),
        round(ev[2] * rng.uniform(0.0, 2.2) * battles),
        round(per_battle_def * rng.uniform(0.0, 2.2) * battles),
        rng.randint(0, battles),
    ]
    return stats, ev


def build():
    rng = random.Random(SEED)
    cases = [{"name": name, "stats": stats, "ev": ev, "expected": python_wn8(stats, ev)}
             for name, stats, ev in EDGE_CASES]

    tank_stats, expected_values = [], {}
    for i in range(RANDOM_CASES):
        stats, ev = random_tank(rng)
        # 1 in 20 random tanks has no expected values, as some real tanks do.
        if rng.random() < 0.05:
            ev = None
        cases.append({"name": f"random {i + 1}", "stats": stats, "ev": ev,
                      "expected": python_wn8(stats, ev)})
        tid = 1000 + 16 * i
        tank_stats.append({"tank_id": tid, "all": dict(zip(STAT_FIELDS, stats))})
        if ev is not None:
            expected_values[tid] = dict(zip(EV_FIELDS, ev))
    # A tank the API returns with no battles, as it does for unplayed garage tanks.
    tank_stats.append({"tank_id": 999999, "all": dict.fromkeys(STAT_FIELDS, 0)})

    df = build_tank_dataframe(tank_stats, expected_values, {})
    overall = calc_overall_wn8(df)
    accounts = [parity_payload(tank_stats, expected_values, df, overall, goal)
                for goal in (overall + 250.0, overall - 250.0)]
    return {"cases": cases, "accounts": accounts}


def render(data):
    """One case per line, so a change to the Python shows up as a readable diff."""
    lines = ["{",
             '"note": "Generated by tests/make_parity_fixture.py. Do not edit by hand.",',
             '"cases": [']
    n = len(data["cases"])
    lines += [json.dumps(c) + ("," if i < n - 1 else "") for i, c in enumerate(data["cases"])]
    lines += ["],", '"accounts": [']
    n = len(data["accounts"])
    lines += [json.dumps(a) + ("," if i < n - 1 else "") for i, a in enumerate(data["accounts"])]
    lines += ["]", "}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if tests/parity_cases.json is out of date")
    args = ap.parse_args()

    text = render(build())
    if args.check:
        current = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if current != text:
            sys.exit(f"{OUT} is out of date: run python tests/make_parity_fixture.py")
        print(f"{OUT} is up to date.")
        return
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    data = json.loads(text)
    print(f"Wrote {OUT}: {len(data['cases'])} cases, {len(data['accounts'])} accounts "
          f"(overall WN8 {data['accounts'][0]['expected']['overall']:.1f}).")


if __name__ == "__main__":
    main()
