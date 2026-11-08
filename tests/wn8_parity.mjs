// Checks that site_assets/wn8.js gives the same WN8 as the Python code.
//
//   node tests/wn8_parity.mjs                     the fixed cases in tests/parity_cases.json
//   node tests/wn8_parity.mjs build/parity.json   a dump from export_player_html.py --dump-parity
//
// Several files can be given at once. Exits 1 if any result differs.
// Each tank's WN8 must agree within 1e-9 (the code does the same operations in
// the same order, so it normally agrees exactly). Account totals must agree
// within a relative 1e-9, because pandas adds the tanks up in a different order.
import { readFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const wn8Path = join(here, "..", "site_assets", "wn8.js");
vm.runInThisContext(readFileSync(wn8Path, "utf8"), { filename: wn8Path });
const WN8 = globalThis.WN8;

const TOL = 1e-9;
const failures = [];
let checks = 0;
let largestTankDiff = 0;

function sameTankWn8(where, got, want) {
  checks++;
  if (got === null || want === null) {
    if (got !== want) failures.push(`${where}: got ${got}, want ${want}`);
    return;
  }
  const diff = Math.abs(got - want);
  largestTankDiff = Math.max(largestTankDiff, diff);
  if (!(diff <= TOL)) failures.push(`${where}: got ${got}, want ${want} (off by ${diff})`);
}

function sameTotal(where, got, want) {
  checks++;
  const diff = Math.abs(got - want);
  if (!(diff <= TOL * Math.max(1, Math.abs(want)))) {
    failures.push(`${where}: got ${got}, want ${want} (off by ${diff})`);
  }
}

function checkCases(cases, label) {
  for (const c of cases) {
    sameTankWn8(`${label}, case "${c.name}"`, WN8.tankWn8(c.stats, c.ev), c.expected);
  }
}

// An account in --dump-parity form: {goal, tanks, ev, expected}.
function checkAccount(acc, label) {
  for (const [id, want] of Object.entries(acc.expected.per_tank)) {
    sameTankWn8(`${label}, tank ${id}`, WN8.tankWn8(acc.tanks[id], acc.ev[id]), want);
  }
  const t = WN8.totals(acc.tanks, acc.ev);
  const overall = WN8.overall(acc.tanks, acc.ev);
  sameTotal(`${label}, overall WN8`, overall, acc.expected.overall);
  checks++;
  if (t.battles !== acc.expected.wn8_battles) {
    failures.push(`${label}, WN8 battles: got ${t.battles}, want ${acc.expected.wn8_battles}`);
  }
  sameTotal(`${label}, deficit`, WN8.deficit(overall, t.battles, acc.goal), acc.expected.deficit);
}

const files = process.argv.slice(2);
if (files.length === 0) files.push(join(here, "parity_cases.json"));

for (const file of files) {
  const label = basename(file);
  const data = JSON.parse(readFileSync(file, "utf8"));
  let found = false;
  if (Array.isArray(data.cases)) {
    checkCases(data.cases, label);
    found = true;
  }
  if (Array.isArray(data.accounts)) {
    data.accounts.forEach((acc, i) => checkAccount(acc, `${label}, account ${i + 1}`));
    found = true;
  }
  if (data.expected) {
    checkAccount(data, label);
    found = true;
  }
  if (!found) failures.push(`${label}: no cases, accounts or expected results in the file`);
}

console.log(`${checks} checks, largest per-tank WN8 difference ${largestTankDiff}.`);
if (failures.length > 0) {
  console.log(`${failures.length} failed:`);
  for (const msg of failures.slice(0, 20)) console.log(`  ${msg}`);
  if (failures.length > 20) console.log(`  ... and ${failures.length - 20} more`);
  process.exit(1);
}
console.log("All match.");
