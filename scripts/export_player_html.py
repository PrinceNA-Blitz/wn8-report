"""
Generate a self-contained static HTML report for one player.

Usage:
    python scripts/export_player_html.py Prince_NA [--region NA] [--out Prince_NA.html]

Site build (GitHub Pages; docs/github_pages_tracker_plan.md):
    python scripts/export_player_html.py Prince_NA --site --account-id 1009796686
        --out site/index.html --write-snapshot data
"""
import argparse
import html
import json
import math
import os
import sys
import time

import pandas as pd
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_ASSETS = os.path.join(ROOT, "site_assets")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import site_data  # noqa: E402
from api_client import REGION_URLS, _get, get_account_info, get_tank_stats, search_player
from data_loader import DATA_DIR, load_expected_values, load_tankopedia
from newscore_calculator import (
    W_DPB_V4,
    W_SR_V4,
    W_SPB_V4,
    calc_newscore_v4_per_tank,
    calc_overall_newscore_v4,
    newscore_color,
    newscore_label,
)
from ui_components import WN8_BANDS, wn8_color, wn8_label
from wn8_calculator import (
    calc_tank_target_dmg_actual,
    build_tank_dataframe,
    calc_overall_wn8,
    calc_tank_target_stats,
)


def wn8_band(score):
    """(color, label) from the app's canonical WN8_BANDS in ui_components."""
    return wn8_color(score), wn8_label(score)


def sortval(v):
    """Raw numeric value for the client-side sorter; NaN sinks to the bottom."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{float(v):.4f}"


def fmt(v, nd=0):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.{nd}f}"


def resolve_player(name, base_url):
    results = search_player(name, base_url)
    for r in results:
        if r.get("nickname", "").lower() == name.lower():
            return r
    return results[0] if results else None


TYPE_LABELS = {
    "lightTank":  "Light",
    "mediumTank": "Medium",
    "heavyTank":  "Heavy",
    "AT-SPG":     "TD",
}

STATUS_TAGS = {"unplayed": "<span class=\"tag garage\">no battles</span>"}


def add_missing_tanks(df, tankopedia, expected_values, garage_battles,
                      min_tier=7, include_locked_premiums=False):
    """
    Append tier min_tier+ tanks that build_tank_dataframe() left out.

    Ownership is NOT knowable from this API. tanks/stats returns 380 entries
    for this account, 47 of them with battles == 0, and the in_garage parameter
    is accepted but ignored (in_garage=1 and in_garage=0 return byte-identical
    responses). A tank missing from the response may be unowned, or may be
    sitting in the garage unplayed - there is no way to tell them apart. So
    every tank with no battles gets one honest label: "no battles".

    Premiums absent from the response are skipped by default. That is a noise
    decision, not an ownership claim: 142 of them exist, most out of
    circulation, and listing them all would bury the tech tree. --all-premiums
    includes them.
    """
    known = set(df["tank_id"]) if not df.empty else set()
    rows = []
    for tid, meta in tankopedia.items():
        if meta.get("tier", 0) < min_tier or tid in known:
            continue
        exp = expected_values.get(tid)
        if not exp or not exp.get("expDmg"):
            continue
        prem = bool(meta.get("is_premium"))
        seen = tid in garage_battles  # returned by the API, in any form
        if prem and not seen and not include_locked_premiums:
            continue
        rows.append({
            "tank_id": tid,
            "name":    meta.get("name", f"Unknown #{tid}"),
            "tier":    meta["tier"],
            "nation":  meta.get("nation", "?"),
            "type":    meta.get("type", "?"),
            "is_premium": prem,
            "battles": 0,
            "wins": 0, "damage_dealt": 0, "frags": 0,
            "survived_battles": 0, "spotted": 0, "dropped_capture_points": 0,
            "exp_dmg": exp["expDmg"],
            "wn8": float("nan"),
            "dmg_per_battle": 0.0,
            "wn8_impact": 0.0,
            "status": "unplayed",
        })
    if not rows:
        return df
    out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    out["status"] = out["status"].fillna("played")
    return out


def fetch_top_gun_alpha(base_url):
    """
    {tank_id: (alpha, gun name)} for each tank's top gun, from the live encyclopedia.

    Top gun = the highest-tier gun the tank can mount (ties broken by damage).
    Alpha is the average damage of that gun's standard (first-listed) shell.
    Returns {} if the encyclopedia is unreachable; the column then shows dashes.
    """
    veh = (_get("encyclopedia/vehicles/", {"fields": "guns"}, base_url) or {}).get("data") or {}
    gun_ids = sorted({g for v in veh.values() if v for g in (v.get("guns") or [])})
    guns = {}
    for i in range(0, len(gun_ids), 100):
        chunk = ",".join(str(g) for g in gun_ids[i:i + 100])
        data = (_get("encyclopedia/modules/", {"module_id": chunk}, base_url) or {}).get("data") or {}
        for g in data.get("guns") or []:
            if g.get("shells"):
                guns[g["module_id"]] = (g.get("tier", 0), g["shells"][0]["damage"], g["name"])
    out = {}
    for tid, v in veh.items():
        cands = [guns[g] for g in ((v or {}).get("guns") or []) if g in guns]
        if cands:
            _tier, alpha, name = max(cands)
            out[int(tid)] = (alpha, name)
    return out


def _num(v, digits=1):
    """Round for JSON; None for NaN/missing."""
    try:
        return None if v is None or pd.isna(v) else round(float(v), digits)
    except (TypeError, ValueError):
        return None


def wn8_by_tier(df, tiers=(10, 9, 8, 7), extra=None):
    """
    Per tier: battles, battle-weighted WN8, and battles/tanks per WN8 band.

    extra: {tank_id: {...}} merged into each tank (target damage, shots, impact).
    """
    extra = extra or {}
    played = df[(df["battles"] > 0) & df["wn8"].notna()]
    total = float(played[played["tier"].isin(tiers)]["battles"].sum()) or 1.0
    out = []
    for t in tiers:
        g = played[played["tier"] == t]
        tb = float(g["battles"].sum())
        if not tb:
            continue
        wn8 = float((g["wn8"] * g["battles"]).sum() / tb)
        bands = []
        for lo, hi, color, label in WN8_BANDS:
            s = g[(g["wn8"] >= lo) & (g["wn8"] < hi)].sort_values("battles", ascending=False)
            if s.empty:
                continue
            bands.append({
                "label": label, "range": f"{lo}{'+' if hi > 9999 else f'–{hi - 1}'}",
                "color": color, "battles": int(s["battles"].sum()),
                "pct": float(s["battles"].sum() / tb * 100),
                "tanks": [{"name": str(r["name"]), "battles": int(r["battles"]),
                           "wn8": float(r["wn8"]), "color": wn8_color(r["wn8"]),
                           "cls": str(r.get("type", "?")),
                           "prem": 1 if r.get("is_premium") else 0,
                           "dmg": _num(r.get("dmg_per_battle"), 0),
                           "wr": _num(r.get("winrate", float("nan")) * 100, 2),
                           "frags": _num(r.get("frags_per_battle"), 2),
                           "spots": _num(r.get("spots_per_battle"), 2),
                           "def": _num(r.get("def_per_battle"), 2),
                           "expDmg": _num(r.get("exp_dmg"), 0),
                           **extra.get(int(r["tank_id"]), {})}
                          for _, r in s.iterrows()],
            })
        top = g.sort_values("battles", ascending=False).head(3)
        out.append({"tier": t, "battles": int(tb), "wn8": wn8, "color": wn8_color(wn8),
                    "share": tb / total * 100, "bands": bands,
                    "top": [f"{r['name']} ({int(r['battles']):,})" for _, r in top.iterrows()]})
    return out


# Plain string (not an f-string) so the JS braces need no doubling.
TIER_CHART_JS = r"""
(function () {
  var chart = document.getElementById("tierChart"), wbtn = document.getElementById("tierWeight");
  var bandBar = document.getElementById("bandFilter");
  if (!chart || !TIER_DATA.length) return;
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) {
    return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); }
  function n(v) { return Math.round(v).toLocaleString(); }

  // WN8 band filter: one button per band present, best band first.
  var seen = {}, bandList = [];
  TIER_DATA.forEach(function (t) { t.bands.forEach(function (b) {
    if (!seen[b.label]) { seen[b.label] = b; bandList.push(b); } }); });
  bandList.sort(function (x, y) { return parseInt(y.range) - parseInt(x.range); });
  bandBar.innerHTML = bandList.map(function (b) {
    // Default view: Great and below (under 2450), the tanks still short of Unicum.
    return '<button class="' + (parseInt(b.range) < 2450 ? "on" : "") + '" data-band="' + esc(b.label) + '" style="--bc:' + b.color + '">' +
      esc(b.label) + ' <span>' + esc(b.range) + '</span></button>';
  }).join("");
  var bandBtns = Array.prototype.slice.call(bandBar.querySelectorAll("button"));
  function active() {
    var on = {};
    bandBtns.forEach(function (b) { if (b.classList.contains("on")) on[b.dataset.band] = 1; });
    return on;
  }

  var pbtn = document.getElementById("sortPlayed"), byPlayed = false;
  // Page filters (tier / class / premium), pushed in by the table filter code.
  var F = null;
  function colorOf(v) {
    for (var i = 0; i < WN8_SCALE.length; i++)
      if (v >= WN8_SCALE[i][0] && v < WN8_SCALE[i][1]) return WN8_SCALE[i][2];
    return "#8b93a7";
  }
  function filtered() {
    var tiers = TIER_DATA.filter(function (t) { return !F || F.tiers.indexOf(String(t.tier)) !== -1; })
      .map(function (t) {
        var bands = t.bands.map(function (b) {
          var tanks = b.tanks.filter(function (k) {
            return !F || (F.cls.indexOf(k.cls) !== -1 && F.prem.indexOf(String(k.prem)) !== -1);
          });
          return {label: b.label, range: b.range, color: b.color, tanks: tanks,
                  battles: tanks.reduce(function (a, k) { return a + k.battles; }, 0)};
        }).filter(function (b) { return b.tanks.length; });
        var tb = bands.reduce(function (a, b) { return a + b.battles; }, 0);
        var ws = bands.reduce(function (a, b) { return a + b.tanks.reduce(function (x, k) {
          return x + k.wn8 * k.battles; }, 0); }, 0);
        bands.forEach(function (b) { b.pct = tb ? b.battles / tb * 100 : 0; });
        return {tier: t.tier, battles: tb, wn8: tb ? ws / tb : 0,
                color: colorOf(tb ? ws / tb : 0), bands: bands};
      }).filter(function (t) { return t.battles; });
    var total = tiers.reduce(function (a, t) { return a + t.battles; }, 0) || 1;
    tiers.forEach(function (t) { t.share = t.battles / total * 100; });
    return tiers;
  }
  function render() {
    var on = active();
    var DATA = filtered();
    if (!DATA.length) {
      chart.innerHTML = '<div class="note" style="margin:0">No played tanks match the filters.</div>';
      return;
    }
    // Only visible bands count: their tanks are rescaled to fill the bar.
    var rows = DATA.map(function (t) {
      var bands = t.bands.filter(function (b) { return on[b.label]; }).reverse();
      var vis = bands.reduce(function (a, b) { return a + b.battles; }, 0);
      return {t: t, bands: bands, vis: vis};
    });
    var maxB = Math.max.apply(null, rows.map(function (r) { return r.vis; })) || 1;
    var allVis = rows.reduce(function (a, r) { return a + r.vis; }, 0) || 1;
    SEGS = [];
    chart.innerHTML = rows.map(function (r) {
      var t = r.t, vis = r.vis || 1;
      // Width follows the shown battles; the label is the band's share of the whole tier,
      // so it stays fixed when other WN8 bands are hidden.
      var head = r.bands.map(function (b) {
        var pct = b.battles / vis * 100, ps = b.pct < 0.1 ? "<0.1%" : b.pct.toFixed(1) + "%";
        return '<div class="bhead" style="width:' + pct + '%;color:' + b.color + '" title="' +
          esc(b.label + " (" + b.range + "): " + n(b.battles) + " battles, " +
          b.pct.toFixed(2) + "% of tier") +
          '" data-l="' + esc(b.label + " " + ps) + '" data-p="' + esc(ps) + '"></div>';
      }).join("");
      function seg(k, b, first) {
        SEGS.push({k: k, b: b, tier: t.tier, vis: vis});
        var key = t.tier + "|" + k.name;
        return '<div class="seg' + (first ? ' first' : '') + (key === selKey ? ' sel' : '') +
          '" data-k="' + (SEGS.length - 1) + '" style="width:' +
          (k.battles / vis * 100) + '%;background:' + b.color + '" title="' +
          esc(k.name + " — " + n(k.battles) + " battles, WN8 " + n(k.wn8) + " (" + b.label +
          ", " + (k.battles / vis * 100).toFixed(2) + "% of shown)") +
          '" data-l="' + esc(k.name) + '"></div>';
      }
      var segs;
      if (byPlayed) {
        // Most played first, left to right, bands mixed; each tank keeps its band colour.
        var all = [];
        r.bands.forEach(function (b) { b.tanks.forEach(function (k) { all.push([k, b]); }); });
        all.sort(function (x, y) { return y[0].battles - x[0].battles; });
        segs = all.map(function (p) { return seg(p[0], p[1], false); }).join("");
        head = "";
      } else {
        segs = r.bands.map(function (b) {
          return b.tanks.map(function (k, i) { return seg(k, b, i === 0); }).join("");
        }).join("");
      }
      return '<div class="trow"><div class="tlab"><b>Tier ' + t.tier + '</b> &middot; ' +
        '<b style="color:' + t.color + '">' + n(t.wn8) + '</b><div>' + n(t.battles) +
        ' battles' + (r.vis < t.battles ? '<br>' + n(r.vis) + ' shown' : '') +
        ' &middot; ' + (r.vis / allVis * 100).toFixed(1) + '% of shown</div></div>' +
        '<div class="tbox" data-w="' + (r.vis / maxB * 100) + '"><div class="bheads">' + head +
        '</div><div class="tbar">' + segs + '</div></div></div>';
    }).join("");
    size();
    showInfo();
  }
  function size() {
    var on = wbtn.classList.contains("on");
    chart.querySelectorAll(".tbox").forEach(function (b) {
      b.style.width = on ? b.dataset.w + "%" : "100%"; });
    // label a tank block only when its rendered width fits the text
    chart.querySelectorAll(".seg").forEach(function (s) {
      s.textContent = s.clientWidth >= 30 ? s.dataset.l : "";
    });
    // Band percents always show: the full label if it fits, else just the percent.
    // Percents that would collide with the previous one drop to a second line.
    chart.querySelectorAll(".bheads").forEach(function (row) {
      var ends = [-1e9, -1e9];
      row.querySelectorAll(".bhead").forEach(function (h) {
        h.textContent = h.dataset.l;
        h.classList.remove("low");
        if (h.scrollWidth > h.clientWidth) h.textContent = h.dataset.p;
        var x = h.offsetLeft, w = h.scrollWidth;
        var line = x >= ends[0] ? 0 : (x >= ends[1] ? 1 : (ends[0] <= ends[1] ? 0 : 1));
        if (line) h.classList.add("low");
        ends[line] = x + Math.max(w, h.clientWidth) + 3;
      });
    });
  }

  // Click a tank block to see its stats; click it again to close.
  var SEGS = [], selKey = null, info = document.getElementById("tankInfo");
  function cell(k, v, c) {
    return '<div><div class="k">' + k + '</div><div class="v"' + (c ? ' style="color:' + c + '"' : '') +
      '>' + v + '</div></div>';
  }
  function f(v, d) { return v == null ? "&mdash;" : Number(v).toLocaleString(undefined,
    {minimumFractionDigits: d || 0, maximumFractionDigits: d || 0}); }
  function showInfo() {
    var s = null;
    SEGS.forEach(function (x) { if (x.tier + "|" + x.k.name === selKey) s = x; });
    if (!s) { info.style.display = "none"; info.innerHTML = ""; return; }
    var k = s.k;
    info.style.display = "block";
    info.innerHTML = '<div class="tihead"><b>' + esc(k.name) + '</b> <span>Tier ' + s.tier +
      ' &middot; ' + esc(k.cls) + (k.prem ? ' &middot; premium' : '') + ' &middot; ' +
      '<span style="color:' + s.b.color + '">' + esc(s.b.label) + '</span></span>' +
      '<button class="ticlose" title="Close">&times;</button></div><div class="tigrid">' +
      cell("Battles", f(k.battles)) +
      cell("WN8", f(k.wn8), k.color) +
      cell("Win rate", k.wr == null ? "&mdash;" : f(k.wr, 2) + "%") +
      cell("Damage", f(k.dmg)) +
      cell("Server avg damage", f(k.expDmg)) +
      cell("Exp Damage (to 2,900)", f(k.target)) +
      cell("Delta Damage", f(k.delta), k.delta == null ? null : (k.delta > 0 ? "#dc2626" : "#16a34a")) +
      '<div title="' + esc(k.gun || "") + '">' + '<div class="k">Exp Shots</div><div class="v">' +
        f(k.shots) + '</div></div>' +
      cell("Frags / battle", f(k.frags, 2)) +
      cell("Spots / battle", f(k.spots, 2)) +
      cell("Impact", k.impact == null ? "&mdash;" : (k.impact > 0 ? "+" : "") + f(k.impact, 2),
           k.impact == null ? null : (k.impact > 0 ? "#16a34a" : "#dc2626")) +
      cell("Share of shown", (k.battles / s.vis * 100).toFixed(2) + "%") +
      '</div>';
  }
  chart.addEventListener("click", function (e) {
    var el = e.target.closest(".seg");
    if (!el) return;
    var s = SEGS[+el.dataset.k], key = s.tier + "|" + s.k.name;
    selKey = selKey === key ? null : key;
    chart.querySelectorAll(".seg.sel").forEach(function (x) { x.classList.remove("sel"); });
    if (selKey) el.classList.add("sel");
    showInfo();
  });
  info.addEventListener("click", function (e) {
    if (!e.target.closest(".ticlose")) return;
    selKey = null;
    chart.querySelectorAll(".seg.sel").forEach(function (x) { x.classList.remove("sel"); });
    showInfo();
  });
  bandBtns.forEach(function (b) {
    b.addEventListener("click", function () {
      // same behaviour as the other filters: first click isolates, later clicks add/remove
      if (bandBtns.every(function (o) { return o.classList.contains("on"); })) {
        bandBtns.forEach(function (o) { o.classList.toggle("on", o === b); });
      } else {
        b.classList.toggle("on");
      }
      if (!bandBtns.some(function (o) { return o.classList.contains("on"); })) {
        bandBtns.forEach(function (o) { o.classList.add("on"); });
      }
      publishBands();
      render();
      if (window.applyTableFilters) window.applyTableFilters();
    });
  });
  wbtn.addEventListener("click", function () { wbtn.classList.toggle("on"); size(); });
  pbtn.addEventListener("click", function () {
    byPlayed = pbtn.classList.toggle("on");
    render();
  });
  window.addEventListener("resize", size);
  window.setTierChartFilter = function (f) { F = f; render(); };
  // Selected WN8 ranges for the main table; null when every band is on.
  function publishBands() {
    var on = active();
    window.tierChartBands = bandBtns.every(function (b) { return b.classList.contains("on"); })
      ? null
      : bandList.filter(function (b) { return on[b.label]; }).map(function (b) {
          var lo = parseInt(b.range), m = b.range.match(/–(\d+)/);
          return [lo, m ? parseInt(m[1]) + 1 : Infinity];
        });
  }
  publishBands();
  render();
})();
"""


# Field order of the compact stats and expected-value arrays used by the page's
# JavaScript (site_assets/wn8.js), the snapshots and the parity dump.
STAT_FIELDS = site_data.STAT_FIELDS
EV_FIELDS = site_data.EV_FIELDS


def exp_damage_target(r, goal_wn8):
    """
    The Exp Damage column: damage per battle needed on a tank to reach goal_wn8.
    Played tanks hold the player's own frag / spot / defence / win ratios on
    that tank where they are. Unplayed tanks have no ratios, so they assume
    every stat is equally far above average (calc_tank_target_stats).
    `r` is a report row, or a dict with at least exp_dmg.
    """
    if r.get("status", "played") == "played" and pd.notna(r.get("rFRAG")):
        return calc_tank_target_dmg_actual(r["exp_dmg"], r["rFRAG"], r["rSPOT"],
                                           r["rDEF"], r["rWIN"], goal_wn8)
    return calc_tank_target_stats({"expDmg": r["exp_dmg"], "expFrag": 0.0, "expSpot": 0.0,
                                   "expDef": 0.0, "expWinRate": 0.0}, goal_wn8)["target_dmg"]


def wn8_battles_of(df):
    """Battles on tanks that have a WN8: the weights calc_overall_wn8 uses."""
    if df.empty:
        return 0.0
    return float(df.loc[df["wn8"].notna(), "battles"].sum())


def deficit_to_goal(overall_wn8, wn8_battles, goal_wn8):
    """The "Deficit to goal" card: points short of the goal x WN8 battles."""
    return max(goal_wn8 - overall_wn8, 0) * wn8_battles


def parity_payload(tank_stats, expected_values, df, overall_wn8, goal_wn8):
    """
    WN8 inputs and Python's results in the page's compact array form, so
    tests/wn8_parity.mjs can check site_assets/wn8.js against this code.
    `df` is build_tank_dataframe's output before add_missing_tanks.
    """
    tanks = {}
    for entry in tank_stats:
        tid = entry.get("tank_id")
        if tid is None:
            continue
        a = entry.get("all", {})
        tanks[str(int(tid))] = [int(a.get(k, 0) or 0) for k in STAT_FIELDS]
    ev = {}
    for tid in tanks:
        exp = expected_values.get(int(tid))
        if exp:
            ev[tid] = [float(exp[k]) for k in EV_FIELDS]
    wn8_by_tank = {}
    if not df.empty:
        for tid, wn8 in zip(df["tank_id"], df["wn8"]):
            wn8_by_tank[str(int(tid))] = None if pd.isna(wn8) else float(wn8)
    wn8_battles = wn8_battles_of(df)
    return {
        "goal": goal_wn8,
        "tanks": tanks,
        "ev": ev,
        "expected": {
            "per_tank": {tid: wn8_by_tank.get(tid) for tid in tanks},
            "overall": float(overall_wn8),
            "wn8_battles": wn8_battles,
            "deficit": deficit_to_goal(overall_wn8, wn8_battles, goal_wn8),
        },
    }


# ── Site build (GitHub Pages) ────────────────────────────────────────────────

def resolve_public_app_id(flag_value=None):
    """
    The application id the page's JavaScript uses to call the API live:
    --public-app-id, else WG_PUBLIC_APP_ID, else the untracked file
    .wg_public_app_id. Anyone can read it in the page. It never falls back to
    the private id on its own; using the same id for both has to be chosen by
    putting it in one of those places, and the build says so when it is.
    """
    value = (flag_value or os.environ.get("WG_PUBLIC_APP_ID") or "").strip()
    path = os.path.join(ROOT, ".wg_public_app_id")
    if not value and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            value = f.read().strip()
    if not value:
        return None
    private = os.path.join(ROOT, ".wg_application_id")
    if os.path.exists(private):
        with open(private, encoding="utf-8") as f:
            if f.read().strip() == value:
                print("Note: the page's application id is the same as the private one "
                      "in .wg_application_id, so the page shows it to anyone.")
    return value


def site_targets(df, expected_values, goal_wn8):
    """Exp Damage for every tank with expected values, as the report works it out."""
    played = {}
    if not df.empty:
        for _, r in df.iterrows():
            if pd.notna(r.get("exp_dmg")):
                played[int(r["tank_id"])] = r
    out = {}
    for tid, exp in sorted(expected_values.items()):
        if not exp.get("expDmg") or exp["expDmg"] <= 0:
            continue
        r = played.get(int(tid), {"status": "unplayed", "exp_dmg": exp["expDmg"]})
        out[str(tid)] = round(float(exp_damage_target(r, goal_wn8)), 1)
    return out


def site_payload(*, app_id, region, base_url, account_id, nickname, goal, current,
                 expected_values, tankopedia, df, overall, history, tz):
    """The SITE object embedded in the page (plan, section 4)."""
    ev = {str(t): [float(e[k]) for k in EV_FIELDS] for t, e in sorted(expected_values.items())}
    meta = {}
    for t in sorted(set(expected_values) | {int(t) for t in current["tanks"]}):
        m = tankopedia.get(t)
        if m:
            meta[str(t)] = [str(m.get("name") or f"Tank #{t}"), int(m.get("tier") or 0),
                            str(m.get("type") or "?"), bool(m.get("is_premium"))]
    return {
        "app_id": app_id, "region": region, "api_base": base_url,
        "account_id": int(account_id), "nickname": nickname, "goal": goal,
        "generated_at": current["taken_at"], "ev_sha256": current["ev_sha256"], "tz": tz,
        "baseline": {"account": current["account"], "tanks": current["tanks"]},
        "ev": ev, "meta": meta, "target": site_targets(df, expected_values, goal),
        "overall": overall, "history": history,
    }


def site_sections(site):
    """HTML that --site adds to the page: (head, session, history, scripts)."""
    if site is None:
        return "", "", "", ""
    since = datetime.fromtimestamp(site["generated_at"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    live = ("" if site["app_id"] else
            " Live updates are off: no public application id was set for this build.")
    days = [d for d in site["history"] if not d.get("start")]
    if days:
        hist = f"{len(days)} day(s) recorded, {days[0]['d']} to {days[-1]['d']}."
    elif site["history"]:
        hist = (f"Tracking started {site['history'][0]['d']}. "
                "Days appear after the next daily build.")
    else:
        hist = "No history yet: the daily build starts it."
    assets = {}
    for name in ("wn8.js", "session.js", "tracker.css"):
        with open(os.path.join(SITE_ASSETS, name), encoding="utf-8") as f:
            assets[name] = f.read()
        closing = "</style" if name.endswith(".css") else "</script"
        if closing in assets[name].lower():
            raise ValueError(f"site_assets/{name} must not contain '{closing}'")
    data = json.dumps(site, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    head = ('\n<meta name="robots" content="noindex">'
            f"\n<style>\n{assets['tracker.css']}</style>")
    # session.js replaces the note below; it stays as the no-JavaScript fallback.
    session = f"""

  <h2>This session</h2>
  <div id="sessionPanel" class="sess">
    <p class="note" style="margin:0">Baseline: your stats when this page was built ({since}).{live}</p>
  </div>"""
    history = f"""

  <h2>History</h2>
  <div id="historyPanel" class="tchart">
    <p class="note" style="margin:0">{html.escape(hist)}</p>
  </div>"""
    scripts = f"""
<script>var SITE = {data};</script>
<script>{assets['wn8.js']}</script>
<script>{assets['session.js']}</script>"""
    return head, session, history, scripts


def build_html(nickname, region, account_id, info, df, overall_wn8, overall_ns,
               goal_wn8=2900.0, max_wn8=None, alpha=None, site=None):
    alpha = alpha or {}
    battles = info.get("statistics", {}).get("all", {}).get("battles", 0)
    stats = info.get("statistics", {}).get("all", {})
    wins = stats.get("wins", 0)
    wr = wins / battles * 100 if battles else 0.0
    avg_dmg = stats.get("damage_dealt", 0) / battles if battles else 0.0
    survival = stats.get("survived_battles", 0) / battles * 100 if battles else 0.0

    wn8_color, wn8_label = wn8_band(overall_wn8)
    # Points short of the goal, and the same gap as a WN8 x battles total: the
    # surplus over the goal that future battles must add up to.
    deficit = goal_wn8 - overall_wn8
    wn8_battles = wn8_battles_of(df)
    # v4 returns exactly 0 only when no tank could be scored (no battles on a tank with norms).
    if overall_ns > 0:
        ns_color, ns_label, ns_value = newscore_color(overall_ns), newscore_label(overall_ns), fmt(overall_ns)
    else:
        ns_color, ns_label, ns_value = "#9fb6d4", "no tank with population norms", "N/A"

    # Target damage = damage/battle needed on that tank to reach goal_wn8.
    # Delta = current damage minus that target; negative means short of goal.
    df_dmg = df.dropna(subset=["exp_dmg", "dmg_per_battle"]).copy()
    df_dmg = df_dmg[df_dmg["tier"] >= 7]
    if max_wn8 is not None:
        # ~(> max) rather than (<= max) so tanks with no WN8 are kept.
        df_dmg = df_dmg[~(df_dmg["wn8"] > max_wn8)]
    df_dmg["target_dmg"] = df_dmg.apply(lambda r: exp_damage_target(r, goal_wn8), axis=1)
    # Same convention as the app's "Δ Dmg": target − current, so positive means
    # this tank still needs that much more damage per battle to reach the goal.
    df_dmg["delta_dmg"] = df_dmg["target_dmg"] - df_dmg["dmg_per_battle"]
    # Drag = points of overall WN8 this tank costs you, i.e. how much the account
    # average would rise if these battles vanished. Same quantity as the app's
    # "Fix" column: (overall - tank_wn8) * battles / total_battles.
    total_b = float(df["battles"].sum()) or 1.0
    # Impact = points of overall WN8 this tank is worth to you right now.
    # Positive = holding your average up, negative = pulling it down. Summing it
    # across every tank gives 0, since it is measured against that same average.
    df_dmg["impact"] = (df_dmg["wn8"] - overall_wn8) * df_dmg["battles"] / total_b
    df_dmg = df_dmg.sort_values("target_dmg", ascending=True)

    def shots_cell(r):
        # Shots per battle that the target damage works out to at the top gun's alpha.
        a = alpha.get(int(r["tank_id"]))
        if not a or pd.isna(r["target_dmg"]):
            return "<td class='num' data-v=''>&mdash;</td>"
        shots = math.ceil(r["target_dmg"] / a[0])  # a partial shot is still a shot
        tip = html.escape(f"{a[1]} — {a[0]:,.0f} alpha", quote=True)
        return f"<td class='num' data-v='{shots}' title=\"{tip}\">{shots}</td>"

    def damage_rows(frame):
        out = []
        for _, r in frame.iterrows():
            delta = r["delta_dmg"]
            delta_color = "#dc2626" if delta > 0 else "#16a34a"
            tank_wn8 = r["wn8"]
            wn8_c, _lab = wn8_band(tank_wn8)  # handles NaN -> grey/N/A
            status = r.get("status", "played")
            new = status != "played"
            impact_color = "#8b93a7" if new else (
                "#16a34a" if r["impact"] > 0 else "#dc2626")
            if new:
                delta_color = "#8b93a7"
            # data-v carries the raw number so sorting never parses formatted text
            out.append(
                f"<tr data-tier='{int(r['tier'])}'"
                f" data-status='{status}'"
                f" data-class='{html.escape(str(r.get('type', '?')))}'"
                f" data-prem='{1 if r.get('is_premium') else 0}'"
                f" data-wn8='{'' if pd.isna(tank_wn8) else round(float(tank_wn8), 2)}'"
                f"{chr(32) + 'class=' + chr(34) + 'new' + chr(34) if new else ''}>"
                f"<td data-v=\"{html.escape(str(r['name']).lower())}\">"
                f"{html.escape(str(r['name']))}"
                f"{STATUS_TAGS.get(status, '')}</td>"
                f"<td class='num' data-v='{sortval(r['tier'])}'>{int(r['tier'])}</td>"
                f"<td class='num' data-v='{sortval(r['battles'])}'>{fmt(r['battles'])}</td>"
                f"<td class='num' data-v='{sortval(tank_wn8)}'"
                f" style='color:{wn8_c};font-weight:600'>{fmt(tank_wn8)}</td>"
                f"<td class='num' data-v='{sortval(r['dmg_per_battle'])}'>"
                f"{'&mdash;' if new else fmt(r['dmg_per_battle'])}</td>"
                f"<td class='num' data-v='{sortval(r['target_dmg'])}'>"
                f"{fmt(r['target_dmg'])}</td>"
                f"{shots_cell(r)}"
                f"<td class='num' data-v='{sortval(delta)}'"
                f" style='color:{delta_color};font-weight:600'>{fmt(delta)}</td>"
                f"<td class='num' data-v='{sortval(r['impact'])}'"
                f" style='color:{impact_color};font-weight:600'>"
                f"{'&mdash;' if new else format(r['impact'], '+,.2f')}</td>"
                "</tr>"
            )
        return "\n".join(out)

    legend = "\n".join(
        f"<div class='chip' style='background:{color}'>"
        f"<b>{label}</b><span>{lo}{'+' if hi > 9999 else f'–{hi - 1}'}</span></div>"
        for lo, hi, color, label in WN8_BANDS
    )

    tiers_present = sorted({int(t) for t in df_dmg["tier"].unique()})
    counts = df_dmg["status"].value_counts() if "status" in df_dmg else {}
    n_unplayed = int(counts.get("unplayed", 0))
    n_played = int(counts.get("played", 0))
    type_counts = df_dmg["type"].value_counts() if "type" in df_dmg else {}
    types_present = [t for t in TYPE_LABELS if type_counts.get(t, 0)]
    n_prem = int(df_dmg["is_premium"].fillna(False).astype(bool).sum())

    extra = {}
    for _, r in df_dmg.iterrows():
        a = alpha.get(int(r["tank_id"]))
        extra[int(r["tank_id"])] = {
            "target": _num(r["target_dmg"], 0), "delta": _num(r["delta_dmg"], 0),
            "impact": _num(r["impact"], 2),
            "shots": math.ceil(r["target_dmg"] / a[0]) if a and pd.notna(r["target_dmg"]) else None,
            "gun": f"{a[1]} — {a[0]:,.0f} alpha" if a else None,
        }
    tier_data = json.dumps(wn8_by_tier(df, extra=extra)).replace("</", "<\\/")
    wn8_scale = json.dumps([[lo, hi, color] for lo, hi, color, _ in WN8_BANDS])

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    site_head, site_session, site_history, site_scripts = site_sections(site)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">{site_head}
<title>{html.escape(nickname)} — WN8 / NewScore Report</title>
<style>
  :root {{
    color-scheme: dark;
    --bg:#0e1116; --card:#161b22; --text:#e6e9ef; --muted:#8b94a3; --line:#262d38;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:2rem 1rem; background:var(--bg); color:var(--text);
    font:15px/1.5 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif; }}
  .wrap {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{ margin:0 0 .25rem; font-size:1.9rem; }}
  .sub {{ color:var(--muted); margin-bottom:1.5rem; font-size:.9rem; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:.75rem; margin-bottom:2rem; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
    padding:1rem; }}
  .card .k {{ font-size:.75rem; text-transform:uppercase; letter-spacing:.05em;
    color:var(--muted); }}
  .card .v {{ font-size:1.6rem; font-weight:700; margin-top:.25rem; }}
  .card .l {{ font-size:.8rem; color:var(--muted); }}
  h2 {{ font-size:1.1rem; margin:2rem 0 .75rem; }}
  .note {{ color:var(--muted); font-size:.82rem; margin:-.4rem 0 .75rem; }}
  .tiers {{ display:flex; flex-wrap:wrap; gap:.4rem; align-items:center;
    margin-bottom:1.5rem; }}
  .tiers button {{ font:inherit; font-size:.82rem; cursor:pointer; padding:.35rem .8rem;
    border-radius:6px; border:1px solid var(--line); background:var(--card);
    color:var(--muted); }}
  .tiers button.on {{ background:#3972C6; border-color:#3972C6; color:#fff;
    font-weight:600; }}
  .tiers .note {{ margin:0 0 0 .5rem; }}
  .tag.garage {{ color:#e0b64a; border-color:#6b5320; }}
  tr.new td {{ background:rgba(57,114,198,.07); }}
  /* Premium tanks: name in light blue so they read apart from tech tree at a
     glance. Scoped to the name cell only, so the WN8/delta/impact columns keep
     their own semantic colours. */
  tr[data-prem="1"] td:first-child {{ color:#6fc7ff; }}
  .tag {{ margin-left:.5rem; font-size:.68rem; letter-spacing:.03em;
         text-transform:uppercase; color:#9fb6d4; border:1px solid #35507a;
         border-radius:4px; padding:.05rem .35rem; vertical-align:middle; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:.35rem; margin-bottom:1.5rem; }}
  .chip {{ border-radius:6px; padding:.35rem .6rem; color:#fff; font-size:.72rem;
    line-height:1.25; text-shadow:0 1px 2px rgba(0,0,0,.45); }}
  .chip b {{ display:block; font-weight:600; }}
  .chip span {{ opacity:.85; font-variant-numeric:tabular-nums; }}
  .tablewrap {{ overflow-x:auto; background:var(--card); border:1px solid var(--line);
    border-radius:10px; }}
  table {{ border-collapse:collapse; width:100%; font-size:.9rem; }}
  th, td {{ padding:.55rem .75rem; text-align:left; border-bottom:1px solid var(--line);
    white-space:nowrap; }}
  th {{ font-size:.75rem; text-transform:uppercase; letter-spacing:.04em;
    color:var(--muted); font-weight:600; cursor:pointer; user-select:none;
    position:sticky; top:0; background:var(--card); }}
  th:hover {{ color:var(--text); }}
  th::after {{ content:"\\2195"; opacity:.3; margin-left:.35rem; }}
  th.sorted {{ color:var(--text); }}
  th.sorted::after {{ content:"\\25BC"; opacity:.9; }}
  th.sorted[data-dir="asc"]::after {{ content:"\\25B2"; }}
  tr:last-child td {{ border-bottom:none; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .tchart {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
    padding:.75rem 1rem; }}
  .trow {{ display:grid; grid-template-columns:150px minmax(0,1fr); gap:.75rem; align-items:center;
    padding:.4rem 0; border-bottom:1px solid var(--line); }}
  .trow:last-child {{ border-bottom:none; }}
  .tlab b {{ font-size:1rem; }} .tlab div {{ font-size:.75rem; color:var(--muted); }}
  .tbar {{ display:flex; height:34px; min-width:0; border-radius:5px; overflow:hidden; }}
  .seg {{ display:block; line-height:34px; text-align:center; text-overflow:ellipsis; padding:0; color:#fff; flex:none; min-width:0;
    font-size:.66rem; font-weight:600; cursor:default; overflow:hidden; white-space:nowrap;
    text-shadow:0 1px 2px rgba(0,0,0,.5); box-shadow:inset -1px 0 rgba(0,0,0,.25); }}
  .seg:hover, .seg.sel {{ filter:brightness(1.25); outline:2px solid #fff; outline-offset:-2px; }}
  .tops {{ grid-column:2; font-size:.75rem; color:var(--muted); margin-top:-.15rem; }}
  #bandFilter button.on {{ background:var(--bc); border-color:var(--bc); }}
  #bandFilter button span {{ opacity:.75; font-weight:400; }}
  .bheads {{ display:flex; height:28px; align-items:flex-start; }}
  .bhead {{ font-size:.68rem; font-weight:700; white-space:nowrap; overflow:visible; position:relative;
    flex:none; min-width:0; line-height:13px; }}
  .bhead.low {{ top:13px; }}
  .seg {{ cursor:pointer !important; }}
  #tankInfo {{ display:none; margin-top:.75rem; background:var(--card); border:1px solid var(--line);
    border-radius:10px; padding:.75rem 1rem; }}
  .tihead {{ display:flex; align-items:baseline; gap:.5rem; flex-wrap:wrap; margin-bottom:.6rem; }}
  .tihead b {{ font-size:1.05rem; }} .tihead span {{ color:var(--muted); font-size:.8rem; }}
  .ticlose {{ margin-left:auto; background:none; border:none; color:var(--muted); font-size:1.2rem; cursor:pointer; }}
  .tigrid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(120px, 1fr)); gap:.6rem 1rem; }}
  .tigrid .k {{ font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }}
  .tigrid .v {{ font-size:1.05rem; font-weight:600; font-variant-numeric:tabular-nums; }}
  .seg.first {{ box-shadow:inset 2px 0 rgba(0,0,0,.6), inset -1px 0 rgba(0,0,0,.25); }}
  footer {{ margin-top:2.5rem; color:var(--muted); font-size:.8rem; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(nickname)}</h1>
  <div class="sub">{region} &middot; account {account_id} &middot; generated {generated}</div>

  <div class="cards">
    <div class="card"><div class="k">WN8</div>
      <div class="v" style="color:{wn8_color}">{fmt(overall_wn8)}</div>
      <div class="l">{wn8_label}</div></div>
    <div class="card"><div class="k">Deficit to {fmt(goal_wn8)}</div>
      <div class="v" style="color:{'#16a34a' if deficit <= 0 else '#dc2626'}">{fmt(deficit_to_goal(overall_wn8, wn8_battles, goal_wn8))}</div>
      <div class="l">WN8 &times; battles</div></div>
    <div class="card"><div class="k">NewScore v4</div>
      <div class="v" style="color:{ns_color}">{ns_value}</div>
      <div class="l">{ns_label}</div></div>
    <div class="card"><div class="k">Battles</div>
      <div class="v">{fmt(battles)}</div><div class="l">{len(df)} tanks played</div></div>
    <div class="card"><div class="k">Win rate</div>
      <div class="v">{fmt(wr, 2)}%</div><div class="l">{fmt(wins)} wins</div></div>
    <div class="card"><div class="k">Avg damage</div>
      <div class="v">{fmt(avg_dmg)}</div><div class="l">per battle</div></div>
    <div class="card"><div class="k">Survival</div>
      <div class="v">{fmt(survival, 1)}%</div><div class="l">battles survived</div></div>
  </div>{site_session}

  <h2>WN8 by tier</h2>
  <p class="note">Bar length = the tier's share of battles (its weight in overall WN8);
    segments = share of that tier's battles in each WN8 band. Each block is one tank, sized by its battles; click one for its stats.</p>
  <div class="tiers" id="bandFilter"></div>
  <div class="tiers"><button id="tierWeight" class="on">Show tier weight</button>
    <button id="sortPlayed">Sort by most played</button></div>
  <div id="tierChart" class="tchart"></div>
  <div id="tankInfo"></div>

  <h2>Tier filter</h2>
  <div class="tiers" id="tierFilter">
    <button data-tier="all" class="on">All</button>
    {"".join(f'<button data-tier="{t}" class="on">{t}</button>' for t in tiers_present)}
    <span class="note" id="tierCount"></span>
  </div>

  <h2>Garage filter</h2>
  <div class="tiers" id="playedFilter">
    <button data-status="played" class="on">Played ({n_played})</button>
    <button data-status="unplayed">No battles ({n_unplayed})</button>
  </div>
  <div class="tiers" id="typeFilter">
    {"".join(
        f'<button data-class="{t}" class="on">{TYPE_LABELS[t]}'
        f' ({type_counts.get(t, 0)})</button>' for t in types_present)}
  </div>
  <div class="tiers" id="premFilter">
    <button data-prem="0" class="on">Tech tree</button>
    <button data-prem="1" class="on">Premium ({n_prem})</button>
  </div>

  <h2>Damage vs exp damage for {fmt(goal_wn8)} WN8, by delta</h2>
  <p class="note">{len(df_dmg)} tanks &middot; tier 7+ &middot; {n_unplayed} with no
    battles (the API cannot say which of those you own){
    f" &middot; excludes tanks already above {fmt(max_wn8)} WN8" if max_wn8 else ""}</p>
  <div class="tablewrap"><table>
    <thead><tr>
      <th data-type="text">Tank</th>
      <th class='num'>Tier</th>
      <th class='num'>Battles</th>
      <th class='num'>WN8</th>
      <th class='num'>Damage</th>
      <th class="num sorted" data-dir="asc">Exp Damage</th>
      <th class='num'>Exp Shots</th>
      <th class='num'>Delta Damage</th>
      <th class='num'>Impact</th>
    </tr></thead>
    <tbody>{damage_rows(df_dmg)}</tbody>
  </table></div>{site_history}

  <footer>
    WN8 uses raw tankavg.json expected values and the standard WN8 formula.
    NewScore v4 weights: damage {W_DPB_V4 * 100:.1f}% / survival {W_SR_V4 * 100:.1f}% / spotting {W_SPB_V4 * 100:.1f}%;
    every battle on a tank with population norms counts.
    Data from the Wargaming WoT Blitz API.
  </footer>
</div>
<script>var TIER_DATA = {tier_data}; var WN8_SCALE = {wn8_scale};</script>
<script>{TIER_CHART_JS}</script>
<script>
// ── Tier filter: hides rows in every table by their data-tier attribute ──────
(function () {{
  var bar = document.getElementById("tierFilter");
  if (!bar) return;
  var btns = Array.prototype.slice.call(bar.querySelectorAll("button"));
  var tierBtns = btns.filter(function (b) {{ return b.dataset.tier !== "all"; }});
  var allBtn = bar.querySelector('[data-tier="all"]');
  var out = document.getElementById("tierCount");
  var playedBar = document.getElementById("playedFilter");
  var playedBtns = playedBar
    ? Array.prototype.slice.call(playedBar.querySelectorAll("button")) : [];
  var premBar = document.getElementById("premFilter");
  var premBtns = premBar
    ? Array.prototype.slice.call(premBar.querySelectorAll("button")) : [];
  var typeBar = document.getElementById("typeFilter");
  var typeBtns = typeBar
    ? Array.prototype.slice.call(typeBar.querySelectorAll("button")) : [];

  // WN8 bands picked in the chart's band filter. Tanks with no battles have no
  // WN8 and always pass, so they stay visible (the Garage filter hides them).
  function inBands(v) {{
    var bands = window.tierChartBands;
    if (!bands) return true;
    if (v === "" || v === undefined) return true;
    v = parseFloat(v);
    return bands.some(function (r) {{ return v >= r[0] && v < r[1]; }});
  }}
  function apply() {{
    var on = tierBtns.filter(function (b) {{ return b.classList.contains("on"); }})
                     .map(function (b) {{ return b.dataset.tier; }});
    allBtn.classList.toggle("on", on.length === tierBtns.length);
    var pOn = playedBtns.filter(function (b) {{ return b.classList.contains("on"); }})
                        .map(function (b) {{ return b.dataset.status; }});
    var mOn = premBtns.filter(function (b) {{ return b.classList.contains("on"); }})
                      .map(function (b) {{ return b.dataset.prem; }});
    var cOn = typeBtns.filter(function (b) {{ return b.classList.contains("on"); }})
                      .map(function (b) {{ return b.dataset.class; }});
    var shown = 0;
    document.querySelectorAll("tbody tr").forEach(function (tr) {{
      var keep = on.indexOf(tr.dataset.tier) !== -1
              && pOn.indexOf(tr.dataset.status) !== -1
              && mOn.indexOf(tr.dataset.prem) !== -1
              && cOn.indexOf(tr.dataset.class) !== -1
              && inBands(tr.dataset.wn8);
      tr.style.display = keep ? "" : "none";
      if (keep) shown++;
    }});
    out.textContent = shown + " rows shown";
    // keep the WN8-by-tier chart in step with the tier / class / premium filters
    window.applyTableFilters = apply;
    if (window.setTierChartFilter) window.setTierChartFilter({{tiers: on, cls: cOn, prem: mOn}});
  }}

  allBtn.addEventListener("click", function () {{
    // "All" is a reset, never a toggle-off — clicking it always shows everything
    tierBtns.forEach(function (b) {{ b.classList.add("on"); }});
    apply();
  }});
  [tierBtns, playedBtns, premBtns, typeBtns].forEach(function (group) {{
    group.forEach(function (b) {{
      b.addEventListener("click", function () {{
        if (group.every(function (o) {{ return o.classList.contains("on"); }})) {{
          // whole group showing: a click isolates this one button
          group.forEach(function (o) {{ o.classList.toggle("on", o === b); }});
        }} else {{
          // already filtered: clicks add or remove buttons from the selection
          b.classList.toggle("on");
        }}
        // never leave a group fully off; an empty selection resets that group
        if (!group.some(function (o) {{ return o.classList.contains("on"); }})) {{
          group.forEach(function (o) {{ o.classList.add("on"); }});
        }}
        apply();
      }});
    }});
  }});
  apply();
}})();

document.querySelectorAll("table").forEach(function (table) {{
  var heads = table.tHead.rows[0].cells;
  Array.prototype.forEach.call(heads, function (th, idx) {{
    th.addEventListener("click", function () {{
      // Same column toggles direction; a new column starts descending,
      // except the text column which starts ascending (A→Z).
      var isText = th.dataset.type === "text";
      var dir;
      if (th.classList.contains("sorted")) {{
        dir = th.dataset.dir === "asc" ? "desc" : "asc";
      }} else {{
        dir = isText ? "asc" : "desc";
      }}
      Array.prototype.forEach.call(heads, function (o) {{
        o.classList.remove("sorted");
        delete o.dataset.dir;
      }});
      th.classList.add("sorted");
      th.dataset.dir = dir;

      var body = table.tBodies[0];
      var rows = Array.prototype.slice.call(body.rows);
      var sign = dir === "asc" ? 1 : -1;
      rows.sort(function (a, b) {{
        var x = a.cells[idx].dataset.v, y = b.cells[idx].dataset.v;
        if (isText) return sign * String(x).localeCompare(String(y));
        // blanks (NaN) always sink to the bottom regardless of direction
        if (x === "" && y === "") return 0;
        if (x === "") return 1;
        if (y === "") return -1;
        return sign * (parseFloat(x) - parseFloat(y));
      }});
      rows.forEach(function (r) {{ body.appendChild(r); }});
    }});
  }});
}});
</script>{site_scripts}
</body>
</html>
"""


def load_stats_file(path):
    """
    Read per-tank stats from a local JSON dump in WG `tanks/stats/` shape.

    Stat-site exports are a time series: one row per tank per snapshot, so a
    tank can appear many times and summing the file raw double-counts badly
    (65,415 battles instead of 42,968 on a real export). Collapse to the most
    recent row per tank_id.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):  # tolerate a full API response envelope
        data = raw.get("data", raw)
        raw = next(iter(data.values())) if isinstance(data, dict) else data
    latest = {}
    for entry in raw:
        tid = entry.get("tank_id")
        if tid is None:
            continue
        tid = int(tid)
        prev = latest.get(tid)
        if prev is None or entry.get("last_battle_time", 0) > prev.get("last_battle_time", 0):
            latest[tid] = entry
    return list(latest.values())


def main():
    if hasattr(sys.stdout, "reconfigure"):  # tank names with non-ASCII characters
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("nickname")
    ap.add_argument("--region", default="NA", choices=list(REGION_URLS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--goal", type=float, default=2900.0,
                    help="Goal WN8 used for the exp-damage column (default 2900)")
    ap.add_argument("--all-premiums", action="store_true",
                    help="Also list premium tanks you do not own (default: hidden)")
    ap.add_argument("--max-wn8", type=float, default=None,
                    help="Exclude tanks already above this WN8 (default: no cap)")
    ap.add_argument("--stats-file", default=None,
                    help="Read per-tank stats from a local JSON dump (e.g. a "
                         "stat-site export) instead of the live WG API. The "
                         "account lookup still uses the API for the header.")
    ap.add_argument("--dump-parity", default=None, metavar="PATH",
                    help="Also write the WN8 inputs and Python's results as JSON, "
                         "for checking site_assets/wn8.js with tests/wn8_parity.mjs")
    ap.add_argument("--site", action="store_true",
                    help="Build the GitHub Pages version: adds the page data, the Session "
                         "and History sections, and asks search engines not to index it")
    ap.add_argument("--account-id", type=int, default=None,
                    help="Look the account up by id instead of searching for the nickname")
    ap.add_argument("--public-app-id", default=None,
                    help="Application id the page uses for live updates (else env "
                         "WG_PUBLIC_APP_ID, else .wg_public_app_id); never falls back "
                         "to the private id")
    ap.add_argument("--write-snapshot", default=None, metavar="DIR",
                    help="Record this build in DIR (initial.json, latest.json, history.jsonl). "
                         "Stops if any stat went down, unless ALLOW_DECREASE=1.")
    ap.add_argument("--history-tz", default="America/Chicago",
                    help="Time zone for the history's day labels (default America/Chicago)")
    args = ap.parse_args()
    if args.write_snapshot and args.stats_file:
        ap.error("--write-snapshot needs live data, so it cannot be used with --stats-file")
    try:
        site_data.day_label(0, args.history_tz)
    except Exception:  # noqa: BLE001  (ZoneInfoNotFoundError, ValueError)
        ap.error(f"unknown time zone {args.history_tz!r}")

    base_url = REGION_URLS[args.region]
    if args.account_id:
        account_id = args.account_id
        info = get_account_info(account_id, base_url)
        if not info:
            sys.exit(f"No account {account_id} found on {args.region}, "
                     "or the API call failed (see any error above).")
        nickname = info.get("nickname") or args.nickname
    else:
        player = resolve_player(args.nickname, base_url)
        if not player:
            sys.exit(f"No player named {args.nickname!r} found on {args.region}.")

        account_id = player["account_id"]
        nickname = player["nickname"]
        info = get_account_info(account_id, base_url)
    if args.stats_file:
        tank_stats = load_stats_file(args.stats_file)
        print(f"Using {len(tank_stats)} tanks from {args.stats_file}")
    else:
        tank_stats = get_tank_stats(account_id, base_url)
    taken_at = int(time.time())
    if not tank_stats:
        sys.exit(f"No tank stats returned for {nickname}: a private profile, "
                 "or the API call failed (see any error above).")

    expected_values = load_expected_values()
    tankopedia = dict(load_tankopedia())
    # The local tankopedia files go stale when WG moves tanks between tiers
    # (e.g. T-34-3 went 8 -> 9), so overlay the live encyclopedia when reachable.
    live = _get("encyclopedia/vehicles/", {"fields": "tier,name,is_premium,type"},
                base_url) or {}
    for tid, v in (live.get("data") or {}).items():
        if v:
            tankopedia[int(tid)] = {**tankopedia.get(int(tid), {}),
                                    **{k: v[k] for k in ("tier", "name", "is_premium", "type")}}
    df = build_tank_dataframe(tank_stats, expected_values, tankopedia)
    df = calc_newscore_v4_per_tank(df)
    # Overall stats must be computed before the unplayed tanks are added, so
    # that 0-battle rows cannot influence the account averages.
    overall_wn8 = calc_overall_wn8(df)
    overall_ns = calc_overall_newscore_v4(df)
    if args.dump_parity:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_parity)), exist_ok=True)
        with open(args.dump_parity, "w", encoding="utf-8") as f:
            json.dump(parity_payload(tank_stats, expected_values, df, overall_wn8, args.goal), f)

    # Site build: work out the snapshot update in memory first, so a problem
    # stops the run before anything is written.
    site, update = None, None
    if args.site or args.write_snapshot:
        current = site_data.make_snapshot(
            taken_at, account_id, site_data.file_sha256(os.path.join(DATA_DIR, "tankavg.json")),
            site_data.account_from_info(info), site_data.tanks_from_api(tank_stats))
        wn8_battles = wn8_battles_of(df)
        overall = {"wn8": float(overall_wn8), "wn8_battles": wn8_battles,
                   "deficit": deficit_to_goal(overall_wn8, wn8_battles, args.goal),
                   "newscore_v4": float(overall_ns)}
    if args.write_snapshot:
        try:
            update = site_data.plan_update(args.write_snapshot, current, overall,
                                           allow_decrease=os.environ.get("ALLOW_DECREASE") == "1")
        except site_data.SnapshotError as e:
            sys.exit(f"Snapshot not written and report not built: {e}")
    if args.site:
        app_id = resolve_public_app_id(args.public_app_id)
        if not app_id:
            print("Note: no public application id, so the page's live updates are off.")
        history = site_data.daily_history(update.lines if update else [], expected_values,
                                          tankopedia, args.history_tz)
        # The snapshot's latest keeps any tank the API left out, so the page's
        # session baseline matches what the next build will compare against.
        site = site_payload(app_id=app_id, region=args.region, base_url=base_url,
                            account_id=account_id, nickname=nickname, goal=args.goal,
                            current=update.latest if update else current,
                            expected_values=expected_values,
                            tankopedia=tankopedia, df=df, overall=overall,
                            history=history, tz=args.history_tz)

    garage_battles = {
        e["tank_id"]: e.get("all", {}).get("battles", 0)
        for e in tank_stats if e.get("tank_id") is not None
    }
    df = add_missing_tanks(df, tankopedia, expected_values, garage_battles,
                           include_locked_premiums=args.all_premiums)

    out = args.out or f"{nickname}.html"
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_html(nickname, args.region, account_id, info, df,
                           overall_wn8, overall_ns, args.goal, args.max_wn8,
                           alpha=fetch_top_gun_alpha(base_url), site=site))
    print(f"Wrote {out}  (WN8 {overall_wn8:.0f}, NewScore {overall_ns:.0f}, "
          f"{len(df)} tanks)")
    if update:
        site_data.write_update(update)
        print(f"Snapshot in {args.write_snapshot}: {update.summary()}")


if __name__ == "__main__":
    main()
