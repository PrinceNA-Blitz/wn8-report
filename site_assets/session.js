/*
 * Live session panel (docs/github_pages_tracker_plan.md, section 5).
 *
 * Shows the battles played since a baseline: the stats embedded at the last
 * daily build, or a reset made in this browser. While the page is visible it
 * asks the Wargaming API every 30 seconds whether anything changed (one small
 * request), and fetches the tank stats only when it did.
 *
 * Needs SITE (embedded by export_player_html.py --site), WN8 (wn8.js) and
 * WN8_SCALE (the report's band colours). Builds the panel with DOM calls only.
 */
(function () {
  "use strict";

  const panel = document.getElementById("sessionPanel");
  if (!panel || typeof SITE === "undefined" || typeof WN8 === "undefined") return;

  const POLL_MS = 30000;
  const TIMEOUT_MS = 15000;
  const BACKOFF_S = [30, 60, 120, 300];
  const STALE_S = 36 * 3600;
  const RESET_KEY = "wn8site:reset:v1";
  const ZERO = WN8.STAT_FIELDS.map(() => 0);
  const TANK_FIELDS = ["tank_id", "last_battle_time"]
    .concat(WN8.STAT_FIELDS.map((f) => "all." + f)).join(",");
  const ACCOUNT = String(SITE.account_id);

  // ── Browser storage (only for the manual reset) ───────────────────────────
  const store = (function () {
    try {
      const s = window.localStorage;
      s.setItem(RESET_KEY + ":probe", "1");
      s.removeItem(RESET_KEY + ":probe");
      return s;
    } catch (e) {
      return null;
    }
  })();

  function loadReset() {
    if (!store) return null;
    try {
      const r = JSON.parse(store.getItem(RESET_KEY));
      // A newer daily build replaces an older reset.
      if (r && r.at > SITE.generated_at && r.tanks) return r;
      if (r) store.removeItem(RESET_KEY);
    } catch (e) { /* unreadable: ignore it */ }
    return null;
  }

  function saveReset(r) {
    try {
      if (r) store.setItem(RESET_KEY, JSON.stringify(r));
      else store.removeItem(RESET_KEY);
    } catch (e) { /* full or blocked: the reset only lasts this visit */ }
  }

  const state = {
    reset: loadReset(),
    current: null,        // {tank id: stats array} from the last fetch
    lastBattle: {},       // {tank id: unix time of its last battle}
    accountKey: null,     // what account/info said last time
    checkedAt: null,      // ms, last successful check
    error: null,
    failures: 0,
    timer: null,
    busy: false,
  };

  function baseline() {
    if (state.reset) return { at: state.reset.at, tanks: state.reset.tanks, manual: true };
    return { at: SITE.generated_at, tanks: SITE.baseline.tanks, manual: false };
  }

  // ── Wargaming API ─────────────────────────────────────────────────────────
  async function api(path, params) {
    const url = new URL(SITE.api_base + path);
    url.searchParams.set("application_id", SITE.app_id);
    for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    try {
      const resp = await fetch(url, { signal: ctrl.signal, cache: "no-store" });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const body = await resp.json();
      if (body.status !== "ok") {
        throw new Error((body.error && body.error.message) || "the API returned an error");
      }
      return body.data;
    } catch (e) {
      if (e.name === "AbortError") throw new Error("no answer within 15 seconds");
      throw e;
    } finally {
      clearTimeout(timer);
    }
  }

  async function check(force) {
    if (!SITE.app_id) return;
    if (state.busy) {               // a Refresh during a check runs right after it
      if (force) state.forceNext = true;
      return;
    }
    state.busy = true;
    clearTimeout(state.timer);
    try {
      const info = (await api("account/info/", {
        account_id: ACCOUNT, fields: "last_battle_time,statistics.all.battles",
      }))[ACCOUNT];
      if (!info) throw new Error("account not found");
      const all = (info.statistics && info.statistics.all) || {};
      const key = info.last_battle_time + ":" + all.battles;
      if (force || !state.current || key !== state.accountKey) {
        const rows = (await api("tanks/stats/", { account_id: ACCOUNT, fields: TANK_FIELDS }))[ACCOUNT];
        if (!rows || !rows.length) throw new Error("no tank stats returned");
        const tanks = {};
        const last = {};
        for (const r of rows) {
          const a = r.all || {};
          tanks[String(r.tank_id)] = WN8.STAT_FIELDS.map((f) => Number(a[f]) || 0);
          last[String(r.tank_id)] = r.last_battle_time || 0;
        }
        state.current = tanks;
        state.lastBattle = last;
        state.accountKey = key;   // only once the tank stats arrived
      }
      state.checkedAt = Date.now();
      state.error = null;
      state.failures = 0;
    } catch (e) {
      state.error = (e && e.message) || String(e);
      state.failures++;
    } finally {
      state.busy = false;
      render();
      schedule();
      if (state.forceNext) {
        state.forceNext = false;
        check(true);
      }
    }
  }

  function retryDelayS() {
    return BACKOFF_S[Math.min(state.failures, BACKOFF_S.length) - 1];
  }

  function schedule() {
    clearTimeout(state.timer);
    if (!SITE.app_id || document.visibilityState !== "visible") return;
    const ms = state.error ? retryDelayS() * 1000 : POLL_MS;
    state.timer = setTimeout(() => check(false), ms);
  }

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") check(false);
    else clearTimeout(state.timer);
  });
  window.addEventListener("online", () => check(false));

  // ── Numbers ───────────────────────────────────────────────────────────────
  // Each tank's term in the overall WN8: WN8 x battles, and its battles, or
  // nothing when it has no WN8 (the same rule as calc_overall_wn8).
  function term(stats, ev) {
    const w = WN8.tankWn8(stats, ev);
    return w === null ? [0, 0] : [w * stats[0], stats[0]];
  }

  function compute() {
    const base = baseline();
    const ev = SITE.ev;
    const goal = SITE.goal;
    const before = WN8.totals(base.tanks, ev);
    const ovBefore = before.battles ? before.weighted / before.battles : 0;
    const defBefore = WN8.deficit(ovBefore, before.battles, goal);
    const out = { base: base, ovBefore: ovBefore, defBefore: defBefore, rows: [], flagged: [],
                  session: {}, ovAfter: ovBefore, defAfter: defBefore,
                  battlesAfter: before.battles };
    if (!state.current) return out;

    let weighted = before.weighted;
    let battles = before.battles;
    const ids = new Set(Object.keys(state.current).concat(Object.keys(base.tanks)));
    for (const id of ids) {
      const b = base.tanks[id] || ZERO;
      const c = state.current[id] || ZERO;
      const d = c.map((v, i) => v - b[i]);
      if (d.some((x) => x < 0)) { out.flagged.push(id); continue; }   // a stat went down
      if (d[0] <= 0) continue;
      out.session[id] = d;
      const tb = term(b, ev[id]);
      const ta = term(c, ev[id]);
      weighted += ta[0] - tb[0];
      battles += ta[1] - tb[1];
      // What this tank alone did to the deficit.
      const w1 = before.weighted - tb[0] + ta[0];
      const b1 = before.battles - tb[1] + ta[1];
      const def1 = WN8.deficit(b1 ? w1 / b1 : 0, b1, goal);
      out.rows.push({
        id: id, d: d,
        wn8: WN8.tankWn8(d, ev[id]),
        lifeBefore: WN8.tankWn8(b, ev[id]),
        lifeAfter: WN8.tankWn8(c, ev[id]),
        impact: def1 - defBefore,
        last: state.lastBattle[id] || 0,
      });
    }
    out.rows.sort((x, y) => y.last - x.last);
    out.battlesAfter = battles;
    out.ovAfter = battles ? weighted / battles : 0;
    out.defAfter = WN8.deficit(out.ovAfter, battles, goal);
    return out;
  }

  // The page's own check that its WN8 agrees with the build's Python numbers.
  (function selfCheck() {
    const t = WN8.totals(SITE.baseline.tanks, SITE.ev);
    const ov = t.battles ? t.weighted / t.battles : 0;
    if (Math.abs(ov - SITE.overall.wn8) > 1e-9 * Math.max(1, SITE.overall.wn8)) {
      console.warn("Session panel: WN8 from the embedded stats (" + ov +
                   ") differs from the build's (" + SITE.overall.wn8 + ")");
    }
  })();

  // ── Formatting ────────────────────────────────────────────────────────────
  const nf = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  const GREEN = "#16a34a";
  const RED = "#dc2626";
  const MUTED = "var(--muted)";

  function n0(v) { return v === null || v === undefined || !isFinite(v) ? "—" : nf.format(Math.round(v)); }
  function signed(v) {
    if (v === null || !isFinite(v)) return "—";
    const r = Math.round(v);
    return (r > 0 ? "+" : r < 0 ? "−" : "±") + nf.format(Math.abs(r));
  }
  function pct(part, whole, digits) {
    return whole ? (100 * part / whole).toFixed(digits) + "%" : "—";
  }
  function when(ts) {
    return new Date(ts * 1000).toLocaleString(undefined,
      { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  }
  function clock(ms) {
    return new Date(ms).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit", second: "2-digit" });
  }
  function wn8Color(v) {
    if (v === null || v === undefined || !isFinite(v)) return "#888888";
    for (const band of WN8_SCALE) if (v >= band[0] && v < band[1]) return band[2];
    return WN8_SCALE[WN8_SCALE.length - 1][2];
  }

  function el(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") e.className = v;
      else if (k === "style") e.style.cssText = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    for (const kid of kids) if (kid !== null && kid !== undefined && kid !== false) e.append(kid);
    return e;
  }

  function tile(label, value, color, sub, subColor) {
    return el("div", { class: "card" },
      el("div", { class: "k" }, label),
      el("div", { class: "v", style: color ? "color:" + color : "" }, value),
      sub ? el("div", { class: "l", style: subColor ? "color:" + subColor + ";font-weight:600" : "" }, sub) : null);
  }

  // ── Rendering ─────────────────────────────────────────────────────────────
  function render() {
    const r = compute();
    const kids = [];
    const now = Date.now() / 1000;

    if (!r.base.manual && now - SITE.generated_at > STALE_S) {
      kids.push(el("div", { class: "sess-warn" },
        "The daily build is late: this baseline is from " + when(SITE.generated_at) + "."));
    }

    let battles = 0;
    let wins = 0;
    let dmg = 0;
    for (const id of Object.keys(r.session)) {
      const d = r.session[id];
      battles += d[0]; wins += d[1]; dmg += d[2];
    }
    const sTotals = WN8.totals(r.session, SITE.ev);
    const sWn8 = sTotals.battles ? sTotals.weighted / sTotals.battles : null;
    const dDef = r.defAfter - r.defBefore;
    const dOv = r.ovAfter - r.ovBefore;

    kids.push(el("div", { class: "sess-cards" },
      tile("Battles", n0(battles)),
      tile("Win rate", pct(wins, battles, 1), null, battles ? n0(wins) + " wins" : null),
      tile("Session WN8", n0(sWn8), wn8Color(sWn8)),
      tile("Avg damage", battles ? n0(dmg / battles) : "—", null, battles ? "per battle" : null),
      tile("Deficit to " + nf.format(SITE.goal), n0(r.defAfter), r.defAfter > 0 ? RED : GREEN,
           signed(dDef) + " this session", dDef < 0 ? GREEN : dDef > 0 ? RED : MUTED),
      r.battlesAfter
        ? tile("Overall WN8", r.ovAfter.toFixed(2), wn8Color(r.ovAfter),
               (dOv >= 0 ? "+" : "−") + Math.abs(dOv).toFixed(2) + " this session",
               dOv > 0 ? GREEN : dOv < 0 ? RED : MUTED)
        : tile("Overall WN8", "—")));

    // Baseline, status and buttons
    const bar = el("div", { class: "sess-bar" });
    bar.append(el("span", null, "Since " + when(r.base.at) +
      (r.base.manual ? " (your reset in this browser)" : " (daily build)")));
    if (!SITE.app_id) {
      bar.append(el("span", { class: "sess-err" }, "Live updates are off: no application id was set for this build."));
    } else if (state.error) {
      bar.append(el("span", { class: "sess-err" },
        "Couldn't check the API (" + state.error + "). " +
        (document.visibilityState === "visible"
          ? "Trying again in " + retryDelayS() + " s."
          : "Paused while this tab is hidden.")));
    } else if (state.checkedAt) {
      bar.append(el("span", null, "Checked " + clock(state.checkedAt) + ", every 30 s while open."));
    } else {
      bar.append(el("span", null, "Checking…"));
    }
    if (SITE.app_id) {
      bar.append(el("button", { type: "button", onclick: () => check(true) }, "Refresh"));
    }
    if (store && state.current) {
      bar.append(el("button", {
        type: "button", title: "Count from now, in this browser only",
        onclick: () => {
          state.reset = { at: Math.floor(Date.now() / 1000), tanks: state.current };
          saveReset(state.reset);
          render();
        },
      }, "Reset here"));
    }
    if (r.base.manual) {
      bar.append(el("button", {
        type: "button",
        onclick: () => { state.reset = null; saveReset(null); render(); },
      }, "Back to daily build"));
    }
    kids.push(bar);

    if (r.flagged.length) {
      kids.push(el("div", { class: "sess-warn" },
        r.flagged.length + " tank(s) show lower stats than the baseline, or are missing " +
        "from the API's answer, and are left out."));
    }

    if (!r.rows.length) {
      kids.push(el("div", { class: "sess-empty" },
        state.current ? "No battles since " + when(r.base.at) + "." :
        SITE.app_id ? "" : "Battles played after the baseline appear here."));
    } else {
      kids.push(sessionTable(r.rows));
    }
    panel.replaceChildren(...kids);
  }

  function sessionTable(rows) {
    const heads = ["Tank", "Battles", "Win %", "WN8", "Damage", "Exp Damage", "vs Exp",
                   "Frags", "Spots", "Survived", "Lifetime WN8", "Deficit impact", "Last battle"];
    // Rows go straight into the table, not a <tbody>: the report's filters hide
    // every "tbody tr" on the page, and these rows have no tier or class data.
    const table = el("table", null,
      el("thead", null, el("tr", null, ...heads.map((h, i) => el("th", { class: i ? "num" : "" }, h)))));
    for (const row of rows) {
      const d = row.d;
      const meta = SITE.meta[row.id];
      const name = meta ? meta[0] : "Tank #" + row.id;
      const target = SITE.target[row.id];
      const dpb = d[2] / d[0];
      const vs = target === undefined ? null : dpb - target;
      table.append(el("tr", { "data-prem": meta && meta[3] ? "1" : "0" },
        el("td", null, name, meta ? el("span", { class: "tier" }, "T" + meta[1]) : null),
        el("td", { class: "num" }, n0(d[0])),
        el("td", { class: "num" }, pct(d[1], d[0], 0)),
        el("td", { class: "num", style: "font-weight:600;color:" + wn8Color(row.wn8) }, n0(row.wn8)),
        el("td", { class: "num" }, n0(dpb)),
        el("td", { class: "num" }, target === undefined ? "—" : n0(target)),
        el("td", { class: "num", style: vs === null ? "" : "font-weight:600;color:" + (vs >= 0 ? GREEN : RED) },
           signed(vs)),
        el("td", { class: "num" }, (d[3] / d[0]).toFixed(2)),
        el("td", { class: "num" }, (d[4] / d[0]).toFixed(2)),
        el("td", { class: "num" }, pct(d[6], d[0], 0)),
        el("td", { class: "num" }, n0(row.lifeBefore) + " → " + n0(row.lifeAfter)),
        el("td", { class: "num", style: "font-weight:600;color:" + (row.impact < 0 ? GREEN : row.impact > 0 ? RED : MUTED) },
           signed(row.impact)),
        el("td", { class: "num" }, row.last ? when(row.last) : "—")));
    }
    return el("div", { class: "tablewrap" }, table);
  }

  render();
  if (SITE.app_id && document.visibilityState === "visible") check(false);
})();
