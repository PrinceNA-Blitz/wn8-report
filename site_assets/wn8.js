/*
 * WN8 for the live session panel.
 *
 * A port of calc_tank_wn8 and calc_overall_wn8 in wn8_calculator.py. The
 * operations are the same and in the same order, so each tank's WN8 matches
 * Python exactly. tests/wn8_parity.mjs checks this: run it after any change
 * here or to the Python.
 *
 * Runs as a plain script in the page and under Node. It sets globalThis.WN8.
 *
 *   stats arrays:          [battles, wins, damage_dealt, frags, spotted,
 *                           dropped_capture_points, survived_battles]
 *   expected-value arrays: [expDmg, expFrag, expSpot, expDef, expWinRate]
 */
(function (root) {
  "use strict";

  const STAT_FIELDS = ["battles", "wins", "damage_dealt", "frags", "spotted",
                       "dropped_capture_points", "survived_battles"];
  const EV_FIELDS = ["expDmg", "expFrag", "expSpot", "expDef", "expWinRate"];

  // WN8 of one tank, or null when it has no battles or no usable expected values.
  function tankWn8(s, e) {
    const battles = s[0];
    if (!battles) return null;
    if (!e || !(e[0] > 0 && e[1] > 0 && e[2] > 0 && e[4] > 0)) return null;

    const dmgPb = s[2] / battles;
    const fragPb = s[3] / battles;
    const spotPb = s[4] / battles;
    const defPb = s[5] / battles;
    const wr = s[1] / battles;

    const expDef = e[3] > 0 ? e[3] : 0.001;

    const rDMG = dmgPb / e[0];
    const rFRAG = fragPb / e[1];
    const rSPOT = spotPb / e[2];
    const rDEF = defPb / expDef;
    const rWIN = wr / e[4];

    const dmgC = Math.max(0.0, (rDMG - 0.22) / (1 - 0.22));
    const fragC = Math.max(0.0, Math.min(dmgC + 0.2, (rFRAG - 0.12) / (1 - 0.12)));
    const spotC = Math.max(0.0, Math.min(dmgC + 0.1, (rSPOT - 0.38) / (1 - 0.38)));
    const defC = Math.max(0.0, Math.min(dmgC + 0.1, (rDEF - 0.10) / (1 - 0.10)));
    const winC = Math.max(0.0, (rWIN - 0.71) / (1 - 0.71));

    return (
      980 * dmgC
      + 210 * dmgC * fragC
      + 155 * fragC * spotC
      + 75 * defC * fragC
      + 145 * Math.min(1.8, winC)
    );
  }

  // The two sums behind calc_overall_wn8: WN8 x battles, and battles, over
  // tanks that have a WN8. tanks and ev are objects keyed by tank id.
  function totals(tanks, ev) {
    let weighted = 0;
    let battles = 0;
    for (const id of Object.keys(tanks)) {
      const s = tanks[id];
      const w = tankWn8(s, ev[id]);
      if (w === null) continue;
      weighted += w * s[0];
      battles += s[0];
    }
    return { weighted: weighted, battles: battles };
  }

  // Overall WN8: the battle-weighted mean over tanks that have a WN8.
  function overall(tanks, ev) {
    const t = totals(tanks, ev);
    return t.battles > 0 ? t.weighted / t.battles : 0.0;
  }

  // The report's "Deficit to goal" card: points short of the goal x WN8 battles.
  function deficit(overallWn8, wn8Battles, goal) {
    return Math.max(goal - overallWn8, 0) * wn8Battles;
  }

  root.WN8 = {
    STAT_FIELDS: STAT_FIELDS,
    EV_FIELDS: EV_FIELDS,
    tankWn8: tankWn8,
    totals: totals,
    overall: overall,
    deficit: deficit,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
