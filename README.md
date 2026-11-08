# Prince_NA WN8 report

A daily-built page of my World of Tanks Blitz stats (NA server), with a live
session tracker. It is published with GitHub Pages from this repository.

## What it shows

- Overall WN8, the deficit to 2,900 WN8 (points short × battles), and NewScore v4.
- WN8 by tier, and a table of tier 7+ tanks with the damage per battle each one
  needs to reach 2,900.
- **This session**: battles played since the last daily build, checked against
  the Wargaming API every 30 seconds while the page is open.
- **History**: one entry per day, from the snapshots in `data/`.

## How it is built

The workflow in `.github/workflows/build.yml` runs every day at 09:30 UTC, and on
demand from the Actions tab. It:

1. runs the tests,
2. builds `site/index.html` with `scripts/export_player_html.py` from live
   Wargaming API data,
3. checks that the page's JavaScript WN8 matches the Python on that data,
4. saves the day's snapshot in `data/` and commits it,
5. deploys the page.

If the API is down, or any stat is lower than at the last build, nothing is
committed or deployed and the previous page stays up. A manual run can record
such a drop as an anomaly instead ("allow decrease").

WN8 is the standard formula, with per-tank expected values from `tankavg.json`.

## Notes

- The code is copied here from a private repository by a sync script;
  `SYNC_MANIFEST.json` records the source commit. Changes are made there.
- No license is granted: all rights reserved.
