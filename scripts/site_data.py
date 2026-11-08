"""
Daily snapshots and history for the GitHub Pages site
(docs/github_pages_tracker_plan.md, section 4).

Each site build keeps three files in a data folder:

- initial.json   every tank's stats at the first build; never rewritten
- latest.json    every tank's stats at the newest build; rewritten each build
- history.jsonl  one line per build: what changed since the build before

Replaying initial.json plus every line's "changed" must give latest.json.
Each build checks that, in memory, before it writes anything, so a damaged
file stops the build instead of spreading.

Tank stats are compact arrays in STAT_FIELDS order, keyed by tank id as a
string (the same form the page's JavaScript uses).
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from wn8_calculator import build_tank_dataframe, calc_overall_wn8

STAT_FIELDS = ("battles", "wins", "damage_dealt", "frags", "spotted",
               "dropped_capture_points", "survived_battles")
EV_FIELDS = ("expDmg", "expFrag", "expSpot", "expDef", "expWinRate")
ACCOUNT_FIELDS = ("battles", "wins", "damage_dealt", "survived_battles")
ZERO = (0,) * len(STAT_FIELDS)

INITIAL = "initial.json"
LATEST = "latest.json"
HISTORY = "history.jsonl"


class SnapshotError(Exception):
    """The snapshot folder is inconsistent, or the new data looks wrong."""


# ── Building a snapshot ──────────────────────────────────────────────────────

def tanks_from_api(tank_stats):
    """{tank id: stats array} for every row tanks/stats returned, 0-battle rows included."""
    tanks = {}
    for entry in tank_stats:
        tid = entry.get("tank_id")
        if tid is None:
            continue
        a = entry.get("all", {})
        tanks[str(int(tid))] = [int(a.get(k, 0) or 0) for k in STAT_FIELDS]
    return dict(sorted(tanks.items(), key=lambda kv: int(kv[0])))


def account_from_info(info):
    """Account totals from account/info, plus the time of the last battle."""
    a = info.get("statistics", {}).get("all", {})
    out = {k: int(a.get(k, 0) or 0) for k in ACCOUNT_FIELDS}
    out["last_battle_time"] = int(info.get("last_battle_time", 0) or 0)
    return out


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_snapshot(taken_at, account_id, ev_sha256, account, tanks):
    return {"taken_at": int(taken_at), "account_id": int(account_id),
            "ev_sha256": ev_sha256, "account": account, "tanks": tanks}


# ── Differences and replay ───────────────────────────────────────────────────

def diff_tanks(prev, cur):
    """Per-tank change from prev to cur, only for tanks where something changed."""
    changed = {}
    for tid in sorted(set(prev) | set(cur), key=int):
        p, c = prev.get(tid, ZERO), cur.get(tid, ZERO)
        d = [b - a for a, b in zip(p, c)]
        if any(d):
            changed[tid] = d
    return changed


def decreased(changed):
    """Tanks where any stat went down (a reset, a correction, or a missing tank)."""
    return [tid for tid, d in changed.items() if min(d) < 0]


def replay(initial_tanks, lines):
    """initial plus every line's changes, in order."""
    state = {tid: list(v) for tid, v in initial_tanks.items()}
    for line in lines:
        for tid, d in line.get("changed", {}).items():
            base = state.get(tid, ZERO)
            state[tid] = [a + b for a, b in zip(base, d)]
    return state


def _nonzero(tanks):
    # A row of zeros and a missing row mean the same thing: no battles recorded.
    return {tid: list(v) for tid, v in tanks.items() if any(v)}


def check_replay(initial, lines, latest):
    got, want = _nonzero(replay(initial["tanks"], lines)), _nonzero(latest["tanks"])
    if got != want:
        bad = sorted((set(got) ^ set(want)) | {t for t in set(got) & set(want) if got[t] != want[t]},
                     key=int)
        raise SnapshotError(
            f"replaying {INITIAL} and {HISTORY} does not give {LATEST}: "
            f"{len(bad)} tank(s) differ, e.g. {', '.join(bad[:5])}")


# ── Reading, planning and writing an update ──────────────────────────────────

def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_state(data_dir):
    """(initial, latest, lines); (None, None, []) for an empty or missing folder."""
    paths = {name: os.path.join(data_dir, name) for name in (INITIAL, LATEST, HISTORY)}
    present = {name for name, p in paths.items() if os.path.exists(p)}
    if not present:
        return None, None, []
    if present != set(paths):
        missing = ", ".join(sorted(set(paths) - present))
        raise SnapshotError(f"{data_dir} is incomplete: {missing} missing")
    lines = []
    with open(paths[HISTORY], encoding="utf-8") as f:
        for n, raw in enumerate(f, 1):
            if raw.strip():
                try:
                    lines.append(json.loads(raw))
                except json.JSONDecodeError as e:
                    raise SnapshotError(f"{HISTORY} line {n} is not valid JSON: {e}") from None
    return _read_json(paths[INITIAL]), _read_json(paths[LATEST]), lines


@dataclass
class Update:
    data_dir: str
    first: bool
    initial: dict
    latest: dict
    line: dict
    lines: list = field(default_factory=list)   # every line, the new one last

    def summary(self):
        if self.first:
            return f"first snapshot, {len(self.latest['tanks'])} tanks"
        changed = self.line["changed"]
        battles = sum(d[0] for d in changed.values() if d[0] > 0)
        text = f"{len(changed)} tank(s) changed, {battles} battle(s) since the last build"
        if self.line["anomalies"]:
            text += f", {len(self.line['anomalies'])} recorded as anomalies"
        return text


def plan_update(data_dir, current, overall, allow_decrease=False):
    """
    Work out the new history line and check the result replays, without
    writing anything. `current` is make_snapshot()'s output; `overall` is
    {wn8, wn8_battles, deficit, newscore_v4} for this build.
    """
    initial, latest, lines = load_state(data_dir)
    account = {k: current["account"][k] for k in ACCOUNT_FIELDS}
    if latest is None:
        line = {"taken_at": current["taken_at"], "prev_taken_at": None,
                "ev_sha256": current["ev_sha256"], "overall": overall,
                "account": account, "changed": {}, "anomalies": []}
        check_replay(current, [line], current)
        return Update(data_dir, True, current, current, line, [line])

    if latest.get("account_id") != current["account_id"]:
        raise SnapshotError(f"{LATEST} is for account {latest.get('account_id')}, "
                            f"not {current['account_id']}")
    if current["taken_at"] <= latest["taken_at"]:
        raise SnapshotError(f"this build's data is not newer than {LATEST}")
    try:
        check_replay(initial, lines, latest)   # the folder as it stands
    except SnapshotError as e:
        if lines and _nonzero(replay(initial["tanks"], lines[:-1])) == _nonzero(latest["tanks"]):
            raise SnapshotError(
                f"{e}. The last build seems to have stopped after writing {HISTORY} "
                f"but before {LATEST}: remove the last line of {HISTORY} and run again."
            ) from None
        raise

    # A tank the API left out keeps its last stats. Counting it as zero would
    # make its whole history look like new battles on the day it comes back.
    missing = [t for t, v in latest["tanks"].items() if t not in current["tanks"] and any(v)]
    tanks = dict(current["tanks"])
    for t in missing:
        tanks[t] = list(latest["tanks"][t])
    new_latest = dict(current, tanks=dict(sorted(tanks.items(), key=lambda kv: int(kv[0]))))

    changed = diff_tanks(latest["tanks"], new_latest["tanks"])
    down = decreased(changed)
    if (down or missing) and not allow_decrease:
        found = []
        if down:
            found.append(f"{len(down)} tank(s) have lower stats than at the last build "
                         f"(e.g. {', '.join(down[:5])})")
        if missing:
            found.append(f"{len(missing)} tank(s) are missing from the API's answer "
                         f"(e.g. {', '.join(missing[:5])})")
        raise SnapshotError("; ".join(found) + ". If that is expected, run again with "
                            "ALLOW_DECREASE=1 to record it as an anomaly.")
    line = {"taken_at": current["taken_at"], "prev_taken_at": latest["taken_at"],
            "ev_sha256": current["ev_sha256"], "overall": overall,
            "account": account, "changed": changed,
            "anomalies": [{"tank_id": int(t), "change": changed[t]} for t in down]
                         + [{"tank_id": int(t), "missing": True} for t in missing]}
    all_lines = lines + [line]
    check_replay(initial, all_lines, new_latest)
    return Update(data_dir, False, initial, new_latest, line, all_lines)


def _write_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, separators=(",", ":"))
        f.write("\n")
    os.replace(tmp, path)


def write_update(update):
    """
    initial.json (first build only), then history.jsonl, then latest.json, each
    written to a temporary file and moved into place, so no file is ever half
    written. If a run stops between the last two, the next build says how to fix it.
    """
    os.makedirs(update.data_dir, exist_ok=True)
    if update.first:
        _write_json_atomic(os.path.join(update.data_dir, INITIAL), update.initial)
    hist = os.path.join(update.data_dir, HISTORY)
    old = ""
    if os.path.exists(hist):
        with open(hist, encoding="utf-8") as f:
            old = f.read()
    tmp = hist + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(old + json.dumps(update.line, separators=(",", ":")) + "\n")
    os.replace(tmp, hist)
    _write_json_atomic(os.path.join(update.data_dir, LATEST), update.latest)


# ── Daily history for the page ───────────────────────────────────────────────

DAY_STARTS_AT = 3   # local hour a day begins: battles before 3 a.m. count to the night before


def day_label(ts, tz):
    """
    The day a unix time belongs to in `tz`, where a day runs from 3 a.m. to
    3 a.m. local time. The daily build runs at 3:30 a.m. CST / 4:30 a.m. CDT,
    just after a day starts, and a manual build after midnight still counts to
    the evening before.
    """
    local = datetime.fromtimestamp(ts, ZoneInfo(tz)) - timedelta(hours=DAY_STARTS_AT)
    return local.date().isoformat()


def _period_stats(changed, expected_values, tankopedia):
    """Battles, tanks, WN8, damage and win rate over one period's changes."""
    good = {tid: d for tid, d in changed.items() if d[0] > 0 and min(d) >= 0}
    battles = sum(d[0] for d in good.values())
    if not battles:
        return {"b": 0, "tanks": 0, "wn8": None, "dmg": None, "wr": None}
    rows = [{"tank_id": int(tid), "all": dict(zip(STAT_FIELDS, d))} for tid, d in good.items()]
    df = build_tank_dataframe(rows, expected_values, tankopedia)
    wn8 = calc_overall_wn8(df) if not df.empty and df["wn8"].notna().any() else None
    return {
        "b": battles,
        "tanks": len(good),
        "wn8": None if wn8 is None else round(float(wn8), 2),
        "dmg": round(sum(d[2] for d in good.values()) / battles, 1),
        "wr": round(sum(d[1] for d in good.values()) / battles, 4),
    }


def daily_history(lines, expected_values, tankopedia, tz):
    """
    One entry per day for the page, oldest first.

    A line covers prev_taken_at -> taken_at and is labelled with the day of
    prev_taken_at (day_label: days run 3 a.m. to 3 a.m.), which is the day the
    battles were played. Lines with the same label (extra manual builds) are
    merged by adding up their changes before the day's WN8 is worked out.
    Day WN8 uses the current expected values, so every day is measured the
    same way; "ov" and "def" are the overall WN8 and deficit as each build
    recorded them. The first build gives a "start" entry with no battles.
    """
    entries, group = [], []
    prev_sha = None

    def flush():
        if not group:
            return
        summed = {}
        for line in group:
            for tid, d in line["changed"].items():
                base = summed.get(tid, ZERO)
                summed[tid] = [a + b for a, b in zip(base, d)]
        last = group[-1]
        entries.append({
            "d": day_label(group[0]["prev_taken_at"], tz),
            "from": group[0]["prev_taken_at"], "to": last["taken_at"],
            **_period_stats(summed, expected_values, tankopedia),
            "ov": round(float(last["overall"]["wn8"]), 2),
            "def": round(float(last["overall"]["deficit"])),
            "ev_changed": any(line["_ev_changed"] for line in group),
        })
        group.clear()

    for line in lines:
        ev_changed = prev_sha is not None and line["ev_sha256"] != prev_sha
        prev_sha = line["ev_sha256"]
        if line["prev_taken_at"] is None:
            flush()
            entries.append({
                "d": day_label(line["taken_at"], tz), "start": True,
                "from": None, "to": line["taken_at"],
                "b": 0, "tanks": 0, "wn8": None, "dmg": None, "wr": None,
                "ov": round(float(line["overall"]["wn8"]), 2),
                "def": round(float(line["overall"]["deficit"])),
                "ev_changed": False,
            })
            continue
        label = day_label(line["prev_taken_at"], tz)
        if group and day_label(group[0]["prev_taken_at"], tz) != label:
            flush()
        group.append({**line, "_ev_changed": ev_changed})
    flush()
    return entries
