"""
Tests for scripts/site_data.py (daily snapshots and history for the site).

    python tests/test_site_data.py

No network and no pytest needed; each test uses its own temporary folder.
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import site_data as sd  # noqa: E402
from wn8_calculator import build_tank_dataframe, calc_overall_wn8  # noqa: E402

TZ = "America/Chicago"
EV = {
    1: {"expDmg": 1500.0, "expFrag": 0.8, "expSpot": 1.1, "expDef": 0.6, "expWinRate": 0.52},
    2: {"expDmg": 2200.0, "expFrag": 0.9, "expSpot": 1.0, "expDef": 0.5, "expWinRate": 0.50},
    3: {"expDmg": 900.0, "expFrag": 0.7, "expSpot": 1.4, "expDef": 0.7, "expWinRate": 0.53},
}
OVERALL = {"wn8": 2650.0, "wn8_battles": 100.0, "deficit": 25000.0, "newscore_v4": 770.0}


def ts(y, mo, d, h, mi=0):
    return int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp())


def snap(taken_at, tanks, ev_sha="ev-a", account_id=1009796686):
    account = {"battles": sum(v[0] for v in tanks.values()), "wins": sum(v[1] for v in tanks.values()),
               "damage_dealt": sum(v[2] for v in tanks.values()),
               "survived_battles": sum(v[6] for v in tanks.values()), "last_battle_time": taken_at - 60}
    return sd.make_snapshot(taken_at, account_id, ev_sha, account, tanks)


T0 = {"1": [100, 55, 180000, 90, 120, 40, 50], "2": [40, 22, 95000, 38, 41, 15, 20],
      "4": [0, 0, 0, 0, 0, 0, 0]}


def build(folder, snapshot, allow_decrease=False):
    update = sd.plan_update(folder, snapshot, OVERALL, allow_decrease)
    sd.write_update(update)
    return update


def files(folder):
    out = {}
    for name in (sd.INITIAL, sd.LATEST, sd.HISTORY):
        p = os.path.join(folder, name)
        out[name] = open(p, encoding="utf-8").read() if os.path.exists(p) else None
    return out


# ── Snapshots ────────────────────────────────────────────────────────────────

def test_tanks_from_api_keeps_zero_rows_and_sorts_by_id():
    api = [{"tank_id": 20, "all": {"battles": 3, "wins": 2, "damage_dealt": 5000}},
           {"tank_id": 3, "all": {"battles": 0}},
           {"all": {"battles": 9}}]
    tanks = sd.tanks_from_api(api)
    assert list(tanks) == ["3", "20"], tanks
    assert tanks["3"] == [0] * 7
    assert tanks["20"] == [3, 2, 5000, 0, 0, 0, 0]


def test_first_build_writes_all_three_files():
    with tempfile.TemporaryDirectory() as d:
        u = build(d, snap(ts(2026, 11, 7, 9, 30), T0))
        assert u.first
        initial, latest, lines = sd.load_state(d)
        assert initial == latest
        assert len(lines) == 1 and lines[0]["prev_taken_at"] is None and lines[0]["changed"] == {}


def test_second_build_records_only_what_changed():
    with tempfile.TemporaryDirectory() as d:
        build(d, snap(ts(2026, 11, 7, 9, 30), T0))
        t1 = dict(T0, **{"1": [103, 57, 186000, 93, 124, 41, 52],      # played 3
                         "3": [2, 1, 2500, 1, 3, 0, 1],                 # new tank
                         "5": [0, 0, 0, 0, 0, 0, 0]})                   # new unplayed row
        u = build(d, snap(ts(2026, 11, 8, 9, 30), t1))
        assert not u.first
        assert u.line["changed"] == {"1": [3, 2, 6000, 3, 4, 1, 2], "3": [2, 1, 2500, 1, 3, 0, 1]}
        assert u.line["anomalies"] == []
        initial, latest, lines = sd.load_state(d)
        assert initial["tanks"] == T0, "initial.json must never be rewritten"
        assert latest["tanks"] == t1
        assert len(lines) == 2
        sd.check_replay(initial, lines, latest)


def test_a_drop_in_stats_stops_the_build_and_writes_nothing():
    with tempfile.TemporaryDirectory() as d:
        build(d, snap(ts(2026, 11, 7, 9, 30), T0))
        before = files(d)
        t1 = dict(T0, **{"2": [39, 22, 95000, 38, 41, 15, 20]})
        try:
            sd.plan_update(d, snap(ts(2026, 11, 8, 9, 30), t1), OVERALL)
        except sd.SnapshotError as e:
            assert "ALLOW_DECREASE" in str(e)
        else:
            raise AssertionError("a drop in battles should stop the build")
        assert files(d) == before


def test_a_missing_tank_counts_as_a_drop():
    with tempfile.TemporaryDirectory() as d:
        build(d, snap(ts(2026, 11, 7, 9, 30), T0))
        t1 = {k: v for k, v in T0.items() if k != "2"}
        try:
            sd.plan_update(d, snap(ts(2026, 11, 8, 9, 30), t1), OVERALL)
        except sd.SnapshotError:
            pass
        else:
            raise AssertionError("a tank vanishing from the API should stop the build")


def test_allow_decrease_records_a_drop_as_an_anomaly_and_still_replays():
    with tempfile.TemporaryDirectory() as d:
        build(d, snap(ts(2026, 11, 7, 9, 30), T0))
        t1 = dict(T0, **{"2": [39, 22, 95000, 38, 41, 15, 20]})
        u = build(d, snap(ts(2026, 11, 8, 9, 30), t1), allow_decrease=True)
        assert u.line["anomalies"] == [{"tank_id": 2, "change": [-1, 0, 0, 0, 0, 0, 0]}]
        initial, latest, lines = sd.load_state(d)
        sd.check_replay(initial, lines, latest)
        # and the next normal build works from the new state
        t2 = dict(t1, **{"1": [104, 58, 188000, 94, 125, 41, 53]})
        build(d, snap(ts(2026, 11, 9, 9, 30), t2))


def test_a_missing_tank_is_carried_forward_and_not_counted_when_it_returns():
    with tempfile.TemporaryDirectory() as d:
        build(d, snap(ts(2026, 11, 7, 9, 30), T0))
        t1 = {k: v for k, v in T0.items() if k != "2"}                  # the API leaves tank 2 out
        u = build(d, snap(ts(2026, 11, 8, 9, 30), t1), allow_decrease=True)
        assert u.line["anomalies"] == [{"tank_id": 2, "missing": True}]
        assert "2" not in u.line["changed"]
        _, latest, _ = sd.load_state(d)
        assert latest["tanks"]["2"] == T0["2"], "the last known stats are kept"
        t2 = dict(T0, **{"1": [104, 58, 188000, 94, 125, 41, 53]})      # tank 2 is back, unchanged
        u2 = build(d, snap(ts(2026, 11, 9, 9, 30), t2))                  # no ALLOW_DECREASE needed
        assert "2" not in u2.line["changed"] and u2.line["anomalies"] == []
        initial, latest, lines = sd.load_state(d)
        sd.check_replay(initial, lines, latest)


def test_a_build_stopped_between_the_two_writes_is_explained():
    with tempfile.TemporaryDirectory() as d:
        build(d, snap(ts(2026, 11, 7, 9, 30), T0))
        t1 = dict(T0, **{"1": [101, 56, 182000, 91, 121, 40, 50]})
        u = sd.plan_update(d, snap(ts(2026, 11, 8, 9, 30), t1), OVERALL)
        # history.jsonl gets the new line, but latest.json is never written
        with open(os.path.join(d, sd.HISTORY), "a", encoding="utf-8") as f:
            f.write(json.dumps(u.line) + "\n")
        try:
            sd.plan_update(d, snap(ts(2026, 11, 9, 9, 30), t1), OVERALL)
        except sd.SnapshotError as e:
            assert "remove the last line" in str(e), e
        else:
            raise AssertionError("a half-finished build should stop the next one")


def test_a_damaged_history_file_is_caught():
    with tempfile.TemporaryDirectory() as d:
        build(d, snap(ts(2026, 11, 7, 9, 30), T0))
        build(d, snap(ts(2026, 11, 8, 9, 30), dict(T0, **{"1": [101, 56, 182000, 91, 121, 40, 50]})))
        path = os.path.join(d, sd.HISTORY)
        lines = open(path, encoding="utf-8").read().splitlines()
        damaged = json.loads(lines[1])
        damaged["changed"]["1"][0] += 1
        lines[1] = json.dumps(damaged)
        open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        try:
            sd.plan_update(d, snap(ts(2026, 11, 9, 9, 30), T0 | {"1": [102, 56, 184000, 91, 121, 40, 50]}), OVERALL)
        except sd.SnapshotError as e:
            assert "replaying" in str(e)
        else:
            raise AssertionError("a damaged history.jsonl should stop the build")


def test_an_incomplete_folder_is_refused():
    with tempfile.TemporaryDirectory() as d:
        build(d, snap(ts(2026, 11, 7, 9, 30), T0))
        os.remove(os.path.join(d, sd.INITIAL))
        try:
            sd.load_state(d)
        except sd.SnapshotError as e:
            assert sd.INITIAL in str(e)
        else:
            raise AssertionError("a folder missing initial.json should be refused")


def test_older_data_and_another_account_are_refused():
    with tempfile.TemporaryDirectory() as d:
        build(d, snap(ts(2026, 11, 7, 9, 30), T0))
        for bad, why in ((snap(ts(2026, 11, 7, 9, 30), T0), "not newer"),
                         (snap(ts(2026, 11, 8, 9, 30), T0, account_id=42), "account")):
            try:
                sd.plan_update(d, bad, OVERALL)
            except sd.SnapshotError as e:
                assert why in str(e), e
            else:
                raise AssertionError(f"should refuse: {why}")


# ── Daily history ────────────────────────────────────────────────────────────

def test_day_label_uses_central_time_and_a_3am_day_start():
    # 03:00 UTC on Nov 7 is 21:00 on Nov 6 in Chicago (CST, UTC-6).
    assert sd.day_label(ts(2026, 11, 7, 3, 0), TZ) == "2026-11-06"
    # In summer (CDT, UTC-5) 04:30 UTC is still the previous evening.
    assert sd.day_label(ts(2026, 7, 4, 4, 30), TZ) == "2026-07-03"
    # The daily build at 09:30 UTC starts a new day in both seasons...
    assert sd.day_label(ts(2026, 11, 8, 9, 30), TZ) == "2026-11-08"   # 03:30 CST
    assert sd.day_label(ts(2026, 7, 4, 9, 30), TZ) == "2026-07-04"    # 04:30 CDT
    # ...and 00:10 or 02:50 at night still belongs to the evening before.
    assert sd.day_label(ts(2026, 11, 9, 6, 10), TZ) == "2026-11-08"   # 00:10 CST
    assert sd.day_label(ts(2026, 11, 9, 8, 50), TZ) == "2026-11-08"   # 02:50 CST


def test_a_manual_build_after_midnight_stays_on_the_evening_it_belongs_to():
    a, m, b = ts(2026, 11, 7, 9, 30), ts(2026, 11, 8, 6, 10), ts(2026, 11, 8, 9, 30)
    lines = [line(None, a, {}),
             line(a, m, {"1": [3, 2, 6000, 3, 4, 1, 2]}),   # evening of Nov 7, up to 00:10
             line(m, b, {"1": [2, 1, 3000, 2, 2, 0, 1]})]   # 00:10 to the 03:30 build
    days = sd.daily_history(lines, EV, {}, TZ)
    assert [x["d"] for x in days] == ["2026-11-07", "2026-11-07"], days
    assert days[1]["b"] == 5


def line(prev, taken, changed, ev_sha="ev-a", wn8=2650.0, deficit=25000.0):
    return {"taken_at": taken, "prev_taken_at": prev, "ev_sha256": ev_sha,
            "overall": {"wn8": wn8, "wn8_battles": 100.0, "deficit": deficit, "newscore_v4": 770.0},
            "account": {}, "changed": changed, "anomalies": []}


def test_daily_history_merges_builds_on_the_same_day():
    a, b, c, e = ts(2026, 11, 6, 9, 30), ts(2026, 11, 7, 9, 30), ts(2026, 11, 7, 20, 0), ts(2026, 11, 8, 9, 30)
    lines = [
        line(None, a, {}),
        line(a, b, {"1": [3, 2, 6000, 3, 4, 1, 2]}),                     # played on Nov 6
        line(b, c, {"1": [2, 1, 3000, 2, 2, 0, 1]}),                     # manual build, Nov 7
        line(c, e, {"2": [4, 3, 10000, 4, 4, 2, 2]}, wn8=2651.0, deficit=24800.0),  # also Nov 7
    ]
    days = sd.daily_history(lines, EV, {}, TZ)
    assert [x["d"] for x in days] == ["2026-11-06", "2026-11-06", "2026-11-07"], days
    assert days[0]["start"] is True and days[0]["b"] == 0
    assert days[1]["b"] == 3 and days[1]["tanks"] == 1
    assert days[2]["b"] == 6 and days[2]["tanks"] == 2
    assert days[2]["from"] == b and days[2]["to"] == e
    assert days[2]["ov"] == 2651.0 and days[2]["def"] == 24800


def test_day_wn8_is_calc_overall_wn8_of_the_days_changes():
    a, b = ts(2026, 11, 6, 9, 30), ts(2026, 11, 7, 9, 30)
    changed = {"1": [3, 2, 6000, 3, 4, 1, 2], "2": [4, 3, 10000, 4, 4, 2, 2], "9": [1, 1, 900, 0, 1, 0, 1]}
    days = sd.daily_history([line(None, a, {}), line(a, b, changed)], EV, {}, TZ)
    rows = [{"tank_id": int(t), "all": dict(zip(sd.STAT_FIELDS, v))} for t, v in changed.items()]
    want = calc_overall_wn8(build_tank_dataframe(rows, EV, {}))
    assert days[1]["wn8"] == round(want, 2), (days[1]["wn8"], want)
    assert days[1]["b"] == 8, "battles count every tank, with or without expected values"
    assert days[1]["dmg"] == round(16900 / 8, 1)


def test_days_without_battles_and_negative_changes():
    a, b, c = ts(2026, 11, 6, 9, 30), ts(2026, 11, 7, 9, 30), ts(2026, 11, 8, 9, 30)
    lines = [line(None, a, {}), line(a, b, {}),
             line(b, c, {"1": [2, 1, 3000, 2, 2, 0, 1], "2": [-40, -22, -95000, -38, -41, -15, -20]})]
    days = sd.daily_history(lines, EV, {}, TZ)
    assert days[1]["b"] == 0 and days[1]["wn8"] is None
    assert days[2]["b"] == 2 and days[2]["tanks"] == 1, "a negative change is left out of the day"


def test_ev_changed_marks_the_day_the_file_changed():
    a, b, c = ts(2026, 11, 6, 9, 30), ts(2026, 11, 7, 9, 30), ts(2026, 11, 8, 9, 30)
    lines = [line(None, a, {}), line(a, b, {}, ev_sha="ev-a"), line(b, c, {}, ev_sha="ev-b")]
    days = sd.daily_history(lines, EV, {}, TZ)
    assert [x["ev_changed"] for x in days] == [False, False, True]


# ── Standalone runner (no pytest required) ──────────────────────────────────

def _collect_tests():
    mod = sys.modules[__name__]
    return [getattr(mod, name) for name in dir(mod)
            if name.startswith("test_") and callable(getattr(mod, name))]


if __name__ == "__main__":
    tests = _collect_tests()
    failures = []
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
            failures.append(t.__name__)
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed.")
    if failures:
        print("Failures:", ", ".join(failures))
        raise SystemExit(1)
