import json
import os

try:
    import streamlit as st
except ImportError:  # plain scripts and the GitHub build run without Streamlit
    st = None

# st.cache_resource when Streamlit is installed, otherwise no caching.
_cache_resource = st.cache_resource if st else (lambda f: f)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# The stat site that publishes tankaverages.json uses its raw per-battle means as
# expected values, unscaled. The old 0.9003 factor was compensating for an additive
# (non-standard) WN8 formula; with the standard formula, 1.0 reproduces that site's
# per-tank WN8 to display rounding (research notes, 2026-09-27).
_EV_CORRECTION = 1.0


def derive_expected_values(raw: list, correction: float = _EV_CORRECTION) -> dict:
    """
    Derive per-battle expected values from a parsed tankavg.json payload.

    Split out from the loader so scripts (and the refresh tooling) can derive
    EVs from an arbitrary payload — e.g. to diff an incoming snapshot against
    the installed one — without a second copy of this arithmetic.
    """
    expected = {}
    for entry in raw:
        tank_id = entry.get("tank_id")
        if tank_id is None:
            continue
        a = entry.get("all", {})
        battles = a.get("battles", 0)
        if not battles or battles <= 0:
            continue
        expected[int(tank_id)] = {
            "expDmg":     a.get("damage_dealt", 0) / battles * correction,
            "expFrag":    a.get("frags", 0)         / battles * correction,
            "expSpot":    a.get("spotted", 0)       / battles * correction,
            "expDef":     a.get("dropped_capture_points", 0) / battles * correction,
            "expWinRate": a.get("wins", 0)          / battles * correction,
        }
    return expected


def _load_expected_values(path: str | None = None,
                          correction: float = _EV_CORRECTION) -> dict:
    """Uncached loader — safe to call from plain scripts with no Streamlit runtime."""
    path = path or os.path.join(DATA_DIR, "tankavg.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return derive_expected_values(raw, correction)


@_cache_resource
def load_expected_values() -> dict:
    """
    Load tankavg.json and index by tank_id.
    Derives per-battle expected values from all.* fields, then applies
    a calibration correction so WN8 aligns with stat sites.
    Returns {tank_id (int): {expDmg, expFrag, expSpot, expDef, expWinRate}}
    """
    return _load_expected_values()


def _load_tankopedia() -> dict:
    """Uncached loader — safe to call from plain scripts with no Streamlit runtime."""
    tankopedia = {}

    for filename in ("bs-tankopedia.json", "tankid.json"):
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data = raw.get("data", {})
        for str_id, tank in data.items():
            tid = int(str_id)
            if tid not in tankopedia:
                names = tank.get("names", {})
                name = names.get("en") or tank.get("name", f"Tank #{tid}")
                tankopedia[tid] = {
                    "name":       name,
                    "tier":       tank.get("tier", 0),
                    "nation":     tank.get("nation", "unknown"),
                    "type":       tank.get("type", "unknown"),
                    "is_premium": tank.get("is_premium", False),
                }

    return tankopedia


@_cache_resource
def load_tankopedia() -> dict:
    """
    Load bs-tankopedia.json, falling back to tankid.json for missing tanks.
    Returns {tank_id (int): {name, tier, nation, type, is_premium}}
    """
    return _load_tankopedia()


@_cache_resource
def load_blitzkit_ev() -> dict:
    """
    Load tankavg_blitzkit.json and index by tank_id.
    Returns {tank_id (int): {expDmg, expFrag, expSpot, expDef, expWinRate}}

    mu.* values in the file are already per-battle averages (pre-divided by battles).
    No EV_CORRECTION is applied — BlitzKit uses battle-weighted population means.
    """
    path = os.path.join(DATA_DIR, "tankavg_blitzkit.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    expected = {}
    for entry in raw:
        tank_id = entry.get("tank_id")
        if tank_id is None:
            continue
        expected[int(tank_id)] = {
            "expDmg":     entry.get("expDmg",     0.0),
            "expFrag":    entry.get("expFrag",     0.0),
            "expSpot":    entry.get("expSpot",     0.0),
            "expDef":     entry.get("expDef",      0.0),
            "expWinRate": entry.get("expWinRate",  0.0),
        }
    return expected


@_cache_resource
def load_blitzkit_norm_ev() -> dict:
    """
    Load NewScore normalization EVs from tankavg_blitzkit.json.
    Returns {tank_id (int): {dpbMean, dpbSd, kpbMean, kpbSd, srMean, srSd}}
    """
    path = os.path.join(DATA_DIR, "tankavg_blitzkit.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    norm = {}
    for entry in raw:
        tank_id = entry.get("tank_id")
        if tank_id is None:
            continue
        ms = entry.get("meanSd", {})
        dpb_sd = ms.get("dpbSd", 1) or 1
        kpb_sd = ms.get("kpbSd", 1) or 1
        sr_sd  = ms.get("survivalRateSd", 1) or 1
        norm[int(tank_id)] = {
            "dpbMean": ms.get("dpbMean", 0.0),
            "dpbSd":   dpb_sd,
            "kpbMean": ms.get("kpbMean", 0.0),
            "kpbSd":   kpb_sd,
            "srMean":  ms.get("survivalRateMean", 0.0),
            "srSd":    sr_sd,
        }
    return norm


def get_tank_name(tank_id: int, tankopedia: dict) -> str:
    return tankopedia.get(tank_id, {}).get("name", f"Unknown #{tank_id}")
