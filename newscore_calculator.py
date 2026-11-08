"""
NewScore calculator — per-tank z-scores rolled into a 0–1000 account score.

Every version compares a player's per-battle stats on a tank with the
players of that same tank (means and SDs from tankavg.json) and combines
the z-scores. v4 is canonical; v0–v3 are kept for comparison and are
described oldest first.

NewScore v0 ("Old Score") — damage, kills and survival:
    z_dpb = min((dmg_dealt/battles             - dpbMean)          / dpbSd,          5.0)
    z_kpb = min((frags/battles                 - kpbMean)          / kpbSd,          5.0)
    z_sr  = min((survived_battles/battles*100  - survivalRateMean) / survivalRateSd, 5.0)

    raw        = 0.657 * z_dpb + 0.198 * z_kpb + 0.145 * z_sr
    curved     = sign(raw) * |raw| ** 0.4          # nonlinear expansion of mid-range
    tank_score = max((curved + k), 0) * C
    account_score = Σ(tank_score × battles) / Σ(battles)

Calibration (theoretical ceiling = 1000; k=1.541244 chosen so a 3-player
anchor set spanning the skill range scored intuitively — legacy/unverified,
kept only for historical comparison, see ADVERSARIAL_REVIEW_SUMMARY.md):
    exponent = 0.4
    k = 1.541244
    C = 290.284362    (= 1000 / (5.0**0.4 + k))

NewScore v1 — large-sample OLS weights (n=35,325):
    Weights re-derived by OLS regression on 35,325-player population sample.
    Kills dropped to zero (p=0.20, VIF=10.3 — fully mediated by damage).
    k recalibrated so population p50 = 500 on the norms of the time. Under
    the installed tankavg.json that median is 492 (research/v4_report.md).
    W_DPB_V1 = 0.768 | W_KPB_V1 = 0.000 | W_SR_V1 = 0.232
    K_V1     = 2.10684 | C_V1     = 249.34586

NewScore v2 (Fable) — superseded. v1 weights/calibration + a
linearized "knee" in the power curve to kill the infinite-slope noise
problem at raw=0, + exclusion of small-population tanks:
    W_DPB_V2 = 0.768 | W_KPB_V2 = 0.000 | W_SR_V2 = 0.232   (= v1 weights)
    K_V2     = 2.10684 | C_V2 = 249.34586                    (= v1 calibration)
    KNEE = 0.5 — inside |raw| < KNEE the curve is linear (slope = KNEE**(0.4-1)),
    matching the power curve's value and continuous at the knee. Outside the
    knee, v2 is bit-for-bit identical to v1 (same K, C, weights), so any
    future calibration work on v1 above the knee carries over unchanged.
    Small-population tanks (number_of_players < 200) are excluded from the
    EV table entirely — this also fixes a bug where tanks with missing
    meanSd silently defaulted to mean=0/SD=1 and scored ~950+ regardless of
    real performance (see tank 305 in tankavg.json, ADVERSARIAL_REVIEW_SUMMARY.md #10).

NewScore v3 (Fable) — empirically re-derived from scratch on a 26,753-account
sample (research/v3_account_dataset.csv, built from a 37,056-account research
sample). Every candidate variable (damage, kills, survival, spots, defense
points, K/D ratio) was tested with a pre-registered selection rule (LASSO
1-SE retention, HC3-robust p<0.01, drop-one ΔR²≥0.005, positive coefficient
sign) — see research/v3_variable_importance.md. Kills, K/D, and defense
points were dropped as redundant/negligible; spots/battle was newly added
(nearly orthogonal to damage, r=0.36, and the single largest drop-one ΔR²
of any candidate). Weights fit by OLS on an 80/20 train split; K/C
calibrated so the sample's median account score was 500 on the norms of the
time (514 under the installed tankavg.json):
    W_DPB_V3 = 0.392047 | W_SR_V3 = 0.357241 | W_SPB_V3 = 0.250712
    K_V3     = 2.38248  | C_V3    = 233.31059
    Uses the same Z_CAP/EXPONENT/KNEE curve (curve_v2) as v2 — only the
    weighted variables and K/C differ. See research/v3_fable_report.md for
    the derivation and the formula spec. Its validation figure (Spearman
    0.909 vs v2's 0.879) held out accounts, not battles: the stats and the
    winrate came from the same battles. On disjoint battles v3 scores 0.820
    and v2 0.797 (research/v4_report.md).

NewScore v4 — canonical. Same three per-tank z-scores as v3 (damage,
survival, spots) and, as it turned out, the same weights; what changes is
everything around them:
    composite_tank = W_DPB_V4*z_dpb + W_SR_V4*z_sr + W_SPB_V4*z_spb
    composite_acct = Σ(composite_tank × battles) / Σ(battles)     # every battle counts
    account_score  = 1000 / (1 + exp(-(composite_acct - MU_V4) / S_V4))
    - Weights re-fit with a split-sample IV estimator that only uses
      covariances between disjoint sets of battles, so the same-battle link
      between surviving and winning cannot leak in. The result is within the
      estimate's own uncertainty of v3's weights.
    - The composite is averaged BEFORE it is mapped to 0–1000, so the score
      no longer depends on how performance is spread across tanks.
    - No minimum-battle cliff: every battle on a tank with norms counts. The
      per-tank display score is shrunk toward the account's own level by
      battles / (battles + N0_V4) instead.
    - A smooth logistic scale with two measured anchors (median account 500,
      99th percentile 900) replaces the power curve, whose slope broke by
      2.5x at the knee and whose bottom ~110 points could not be reached.
    On disjoint-battle validation (stats of one half of an account's tanks
    predicting winrate on the other half) v4 scores 0.86 Spearman against
    v3's 0.82, with split-half reliability 0.96 against 0.92 — level with
    WN8 (0.86), which uses winrate as an input. A per-tier adjustment and
    weights fit on tiers 8–10 were tried and rejected after review. Constants
    come from scripts/v4_research.py; see research/v4_report.md.

Tier-based minimum battles per tank (v0–v3 only; v4 has none):
    Tiers 1–4: 50 | Tiers 5–7: 75 | Tiers 8–9: 100 | Tier 10: 150
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ── Calibrated constants ──────────────────────────────────────────────────────
Z_CAP    = 5.0
EXPONENT = 0.4
K        = 1.541244
C        = 290.284362

W_DPB = 0.657
W_KPB = 0.198
W_SR  = 0.145

# ── NewScore v1 — large-sample OLS weights (n=35,325) ────────────────────────
K_V1     = 2.10684
C_V1     = 249.34586

W_DPB_V1 = 0.768
W_KPB_V1 = 0.000
W_SR_V1  = 0.232

# ── NewScore v2 (Fable) — superseded ──────────────────────────────────────────
# v1 weights + calibration (unchanged), plus a linearized knee in the power
# curve to remove the infinite-slope noise blowup at raw=0, plus exclusion of
# small-population tanks (see _load_norm_ev). Identical to v1 for |raw| >= KNEE.
W_DPB_V2 = W_DPB_V1
W_KPB_V2 = W_KPB_V1
W_SR_V2  = W_SR_V1

K_V2 = K_V1
C_V2 = C_V1

KNEE       = 0.5                          # linearization radius around raw=0
KNEE_SLOPE = KNEE ** (EXPONENT - 1.0)     # = 0.5**-0.6 ≈ 1.515717, finite slope at 0

MIN_TANK_POPULATION = 200                 # tanks with fewer sampled players are excluded

# ── NewScore v3 (Fable) — empirically re-derived, superseded by v4 ───────────
# Weights fit by OLS on research/v3_account_dataset.csv (n=26,753 accounts),
# variables chosen by the pre-registered rule in research/v3_variable_importance.md.
# Same curve (curve_v2) as v2; only the weighted variables and K/C differ.
W_DPB_V3 = 0.392047
W_SR_V3  = 0.357241
W_SPB_V3 = 0.250712

K_V3 = 2.38248
C_V3 = 233.31059

TIER_MIN_BATTLES = {
    1: 50, 2: 50, 3: 50, 4: 50,
    5: 75, 6: 75, 7: 75,
    8: 100, 9: 100, 10: 150,
}
DEFAULT_MIN = 75

# ── Score bands (0–1000) ──────────────────────────────────────────────────────
# Shared by every version. Under v4, 500 is the median account of the research
# sample and 900 its 99th percentile; the share of accounts in each band is in
# research/v4_report.md, section 3.6.
NEWSCORE_BANDS = [
    (   0,  100, "#000000", "Consider Another Game"),
    ( 100,  300, "#dc2626", "Really Bad"),
    ( 300,  450, "#ea580c", "Below Average"),
    ( 450,  600, "#16a34a", "Average"),
    ( 600,  750, "#2563eb", "Good"),
    ( 750,  850, "#a855f7", "Really Good"),
    ( 850,  950, "#7e22ce", "Elite"),
    ( 950, 1001, "#3b0764", "Top Dawg"),
]


def newscore_color(score: float) -> str:
    score = max(0.0, min(1000.0, float(score)))
    for lo, hi, color, _ in NEWSCORE_BANDS:
        if lo <= score < hi:
            return color
    return NEWSCORE_BANDS[-1][2]


def newscore_label(score: float) -> str:
    score = max(0.0, min(1000.0, float(score)))
    for lo, hi, _, label in NEWSCORE_BANDS:
        if lo <= score < hi:
            return label
    return NEWSCORE_BANDS[-1][3]


# ── Normalization EV loading ──────────────────────────────────────────────────
_norm_ev: dict | None = None


def _load_norm_ev() -> dict:
    """
    Load per-tank normalization EVs from tankavg.json.

    Entries with missing/empty `meanSd` or `number_of_players` below
    MIN_TANK_POPULATION are skipped entirely (not backfilled with
    mean=0/SD=1). This fixes a bug where such tanks silently scored
    ~950+ regardless of actual performance — see tank 305 (1 player,
    no meanSd) in ADVERSARIAL_REVIEW_SUMMARY.md #10. Skipped tanks flow
    through the existing `tid not in norm_ev -> NaN` path in every scorer.
    """
    global _norm_ev
    if _norm_ev is None:
        path = Path(__file__).parent / "tankavg.json"
        with open(path) as f:
            raw = json.load(f)
        _norm_ev = {}
        for entry in raw:
            tid = entry.get("tank_id")
            if not tid:
                continue
            ms = entry.get("meanSd")
            if not ms:
                continue
            if entry.get("number_of_players", 0) < MIN_TANK_POPULATION:
                continue
            dpb_sd = ms.get("dpbSd", 1) or 1
            kpb_sd = ms.get("kpbSd", 1) or 1
            sr_sd  = ms.get("survivalRateSd", 1) or 1
            _norm_ev[int(tid)] = {
                "dpbMean": ms.get("dpbMean", 0),
                "dpbSd":   dpb_sd,
                "kpbMean": ms.get("kpbMean", 0),
                "kpbSd":   kpb_sd,
                "srMean":  ms.get("survivalRateMean", 0),
                "srSd":    sr_sd,
            }
    return _norm_ev


# ── Per-tank scoring ──────────────────────────────────────────────────────────

def calc_newscore_per_tank(df: pd.DataFrame, norm_ev: dict | None = None) -> pd.DataFrame:
    """
    Add newscore, ns_z_dpb, ns_z_kpb, ns_z_sr columns to a copy of df.

    Requires df columns: tank_id, tier, battles, damage_dealt, frags, survived_battles.
    Tanks below the tier min-battles threshold or without norm EVs get NaN.

    norm_ev: optional pre-loaded dict {tank_id: {dpbMean, dpbSd, kpbMean, kpbSd, srMean, srSd}}.
             Defaults to tankavg.json (BlitzStars) if None.
             Pass load_blitzkit_norm_ev() from data_loader to use BlitzKit EVs.
    """
    if norm_ev is None:
        norm_ev = _load_norm_ev()
    ns_scores, z_dpbs, z_kpbs, z_srs = [], [], [], []

    for _, row in df.iterrows():
        tid     = int(row["tank_id"])
        battles = float(row.get("battles", 0))
        tier    = int(row.get("tier", 0))
        min_b   = TIER_MIN_BATTLES.get(tier, DEFAULT_MIN)

        if battles < min_b or tid not in norm_ev:
            ns_scores.append(float("nan"))
            z_dpbs.append(float("nan"))
            z_kpbs.append(float("nan"))
            z_srs.append(float("nan"))
            continue

        ev  = norm_ev[tid]
        dpb = float(row.get("damage_dealt",     0)) / battles
        kpb = float(row.get("frags",            0)) / battles
        sr  = float(row.get("survived_battles", 0)) / battles * 100.0

        z_dpb = min((dpb - ev["dpbMean"]) / ev["dpbSd"], Z_CAP)
        z_kpb = min((kpb - ev["kpbMean"]) / ev["kpbSd"], Z_CAP)
        z_sr  = min((sr  - ev["srMean"])  / ev["srSd"],  Z_CAP)

        raw    = W_DPB * z_dpb + W_KPB * z_kpb + W_SR * z_sr
        curved = math.copysign(abs(raw) ** EXPONENT, raw)
        score  = max(curved + K, 0.0) * C

        ns_scores.append(round(score, 1))
        z_dpbs.append(round(z_dpb, 3))
        z_kpbs.append(round(z_kpb, 3))
        z_srs.append(round(z_sr,   3))

    result = df.copy()
    result["newscore"] = ns_scores
    result["ns_z_dpb"] = z_dpbs
    result["ns_z_kpb"] = z_kpbs
    result["ns_z_sr"]  = z_srs
    return result


def calc_overall_newscore(df_ns: pd.DataFrame) -> float:
    """Battle-weighted average NewScore across qualifying tanks."""
    valid = df_ns.dropna(subset=["newscore"])
    if valid.empty:
        return 0.0
    return float(
        (valid["newscore"] * valid["battles"]).sum() / valid["battles"].sum()
    )


def calc_newscore_v1_per_tank(df: pd.DataFrame, norm_ev: dict | None = None) -> pd.DataFrame:
    """
    Add newscore_v1 column to a copy of df using large-sample OLS weights.

    Weights re-derived by regression on 35,325-player population sample:
      damage/battle 76.8% · kills/battle 0% · survival rate 23.2%
    k recalibrated so population p50 = 500.

    Requires the same columns as calc_newscore_per_tank.
    norm_ev defaults to tankavg.json if None.
    """
    if norm_ev is None:
        norm_ev = _load_norm_ev()
    ns_scores = []
    ns_counted = []

    for _, row in df.iterrows():
        tid     = int(row["tank_id"])
        battles = float(row.get("battles", 0))
        tier    = int(row.get("tier", 0))
        min_b   = TIER_MIN_BATTLES.get(tier, DEFAULT_MIN)
        qualifies = battles >= min_b and tid in norm_ev

        if battles == 0 or tid not in norm_ev:
            ns_scores.append(float("nan"))
            ns_counted.append(False)
            continue

        ev  = norm_ev[tid]
        dpb = float(row.get("damage_dealt",     0)) / battles
        kpb = float(row.get("frags",            0)) / battles
        sr  = float(row.get("survived_battles", 0)) / battles * 100.0

        z_dpb = min((dpb - ev["dpbMean"]) / ev["dpbSd"], Z_CAP)
        z_kpb = min((kpb - ev["kpbMean"]) / ev["kpbSd"], Z_CAP)
        z_sr  = min((sr  - ev["srMean"])  / ev["srSd"],  Z_CAP)

        raw    = W_DPB_V1 * z_dpb + W_KPB_V1 * z_kpb + W_SR_V1 * z_sr
        curved = math.copysign(abs(raw) ** EXPONENT, raw)
        score  = max(curved + K_V1, 0.0) * C_V1

        ns_scores.append(round(score, 1))
        ns_counted.append(qualifies)

    result = df.copy()
    result["newscore_v1"]       = ns_scores
    result["newscore_v1_counted"] = ns_counted  # False = below min-battles, not in overall
    return result


def calc_overall_newscore_v1(df_ns: pd.DataFrame) -> float:
    """Battle-weighted average NewScore v1 across qualifying tanks (counted only)."""
    if "newscore_v1_counted" in df_ns.columns:
        valid = df_ns[df_ns["newscore_v1_counted"] == True].dropna(subset=["newscore_v1"])
    else:
        valid = df_ns.dropna(subset=["newscore_v1"])
    if valid.empty:
        return 0.0
    return float(
        (valid["newscore_v1"] * valid["battles"]).sum() / valid["battles"].sum()
    )


# ── NewScore v2 (Fable) — linearized-knee power curve ─────────────────────────

def curve_v2(raw: float) -> float:
    """
    Sign-preserving power curve with a linearized knee around raw=0.

    Outside the knee (|raw| >= KNEE) this is identical to the v0/v1 curve:
    sign(raw) * |raw|**EXPONENT. Inside the knee it's linear through the
    origin with slope KNEE_SLOPE = KNEE**(EXPONENT-1), chosen so the line
    meets the power curve exactly at |raw| = KNEE (value-continuous; the
    slope itself changes at the knee). This gives finite slope at raw=0
    instead of the v0/v1 curve's infinite slope there.
    """
    if abs(raw) < KNEE:
        return raw * KNEE_SLOPE
    return math.copysign(abs(raw) ** EXPONENT, raw)


def calc_newscore_v2_per_tank(df: pd.DataFrame, norm_ev: dict | None = None) -> pd.DataFrame:
    """
    Add newscore_v2 column to a copy of df using v1 weights/calibration and
    the linearized-knee curve (curve_v2).

    Weights and K/C are identical to v1 (see module docstring); the only
    difference is the curve shape near raw=0 and the exclusion of
    small-population tanks in norm_ev (see _load_norm_ev).

    Requires the same columns as calc_newscore_per_tank.
    norm_ev defaults to tankavg.json (with small-tank exclusion) if None.
    """
    if norm_ev is None:
        norm_ev = _load_norm_ev()
    ns_scores = []
    ns_counted = []

    for _, row in df.iterrows():
        tid     = int(row["tank_id"])
        battles = float(row.get("battles", 0))
        tier    = int(row.get("tier", 0))
        min_b   = TIER_MIN_BATTLES.get(tier, DEFAULT_MIN)
        qualifies = battles >= min_b and tid in norm_ev

        if battles == 0 or tid not in norm_ev:
            ns_scores.append(float("nan"))
            ns_counted.append(False)
            continue

        ev  = norm_ev[tid]
        dpb = float(row.get("damage_dealt",     0)) / battles
        kpb = float(row.get("frags",            0)) / battles
        sr  = float(row.get("survived_battles", 0)) / battles * 100.0

        z_dpb = max(min((dpb - ev["dpbMean"]) / ev["dpbSd"], Z_CAP), -Z_CAP)
        z_kpb = max(min((kpb - ev["kpbMean"]) / ev["kpbSd"], Z_CAP), -Z_CAP)
        z_sr  = max(min((sr  - ev["srMean"])  / ev["srSd"],  Z_CAP), -Z_CAP)

        raw    = W_DPB_V2 * z_dpb + W_KPB_V2 * z_kpb + W_SR_V2 * z_sr
        curved = curve_v2(raw)
        score  = max(curved + K_V2, 0.0) * C_V2

        ns_scores.append(round(score, 1))
        ns_counted.append(qualifies)

    result = df.copy()
    result["newscore_v2"]         = ns_scores
    result["newscore_v2_counted"] = ns_counted  # False = below min-battles, not in overall
    return result


def calc_overall_newscore_v2(df_ns: pd.DataFrame) -> float:
    """Battle-weighted average NewScore v2 across qualifying tanks (counted only)."""
    if "newscore_v2_counted" in df_ns.columns:
        valid = df_ns[df_ns["newscore_v2_counted"] == True].dropna(subset=["newscore_v2"])
    else:
        valid = df_ns.dropna(subset=["newscore_v2"])
    if valid.empty:
        return 0.0
    return float(
        (valid["newscore_v2"] * valid["battles"]).sum() / valid["battles"].sum()
    )


# ── NewScore v3 (Fable) — empirically re-derived, superseded by v4 ───────────
_norm_ev_v3: dict | None = None


def _load_norm_ev_v3() -> dict:
    """
    Load per-tank normalization EVs for v3 (dpb, sr, spb) from tankavg.json.

    Same exclusion rule as _load_norm_ev (missing meanSd or population below
    MIN_TANK_POPULATION are skipped), plus tanks whose meanSd block is missing
    spbMean/spbSd are also skipped — v3's spots/battle input has no fallback.
    """
    global _norm_ev_v3
    if _norm_ev_v3 is None:
        path = Path(__file__).parent / "tankavg.json"
        with open(path) as f:
            raw = json.load(f)
        _norm_ev_v3 = {}
        for entry in raw:
            tid = entry.get("tank_id")
            if not tid:
                continue
            ms = entry.get("meanSd")
            if not ms:
                continue
            if entry.get("number_of_players", 0) < MIN_TANK_POPULATION:
                continue
            if "spbMean" not in ms or "spbSd" not in ms:
                continue
            dpb_sd = ms.get("dpbSd", 1) or 1
            sr_sd  = ms.get("survivalRateSd", 1) or 1
            spb_sd = ms.get("spbSd", 1) or 1
            _norm_ev_v3[int(tid)] = {
                "dpbMean": ms.get("dpbMean", 0),
                "dpbSd":   dpb_sd,
                "srMean":  ms.get("survivalRateMean", 0),
                "srSd":    sr_sd,
                "spbMean": ms.get("spbMean", 0),
                "spbSd":   spb_sd,
            }
    return _norm_ev_v3


def calc_newscore_v3_per_tank(df: pd.DataFrame, norm_ev: dict | None = None) -> pd.DataFrame:
    """
    Add newscore_v3 column to a copy of df using the empirically re-derived
    v3 ("Fable") weights (damage 39.2% / survival 35.7% / spots 25.1%) and
    the v2 curve/knee machinery. See module docstring and
    research/v3_fable_report.md for the derivation.

    Requires df columns: tank_id, tier, battles, damage_dealt,
    survived_battles, spotted.
    norm_ev defaults to _load_norm_ev_v3() (tankavg.json) if None.
    """
    if norm_ev is None:
        norm_ev = _load_norm_ev_v3()
    ns_scores = []
    ns_counted = []
    z_dpbs, z_srs, z_spbs = [], [], []

    for _, row in df.iterrows():
        tid     = int(row["tank_id"])
        battles = float(row.get("battles", 0))
        tier    = int(row.get("tier", 0))
        min_b   = TIER_MIN_BATTLES.get(tier, DEFAULT_MIN)
        qualifies = battles >= min_b and tid in norm_ev

        if battles == 0 or tid not in norm_ev:
            ns_scores.append(float("nan"))
            ns_counted.append(False)
            z_dpbs.append(float("nan"))
            z_srs.append(float("nan"))
            z_spbs.append(float("nan"))
            continue

        ev  = norm_ev[tid]
        dpb = float(row.get("damage_dealt",     0)) / battles
        sr  = float(row.get("survived_battles", 0)) / battles * 100.0
        spb = float(row.get("spotted",          0)) / battles

        z_dpb = max(min((dpb - ev["dpbMean"]) / ev["dpbSd"], Z_CAP), -Z_CAP)
        z_sr  = max(min((sr  - ev["srMean"])  / ev["srSd"],  Z_CAP), -Z_CAP)
        z_spb = max(min((spb - ev["spbMean"]) / ev["spbSd"], Z_CAP), -Z_CAP)

        raw    = W_DPB_V3 * z_dpb + W_SR_V3 * z_sr + W_SPB_V3 * z_spb
        curved = curve_v2(raw)
        score  = max(curved + K_V3, 0.0) * C_V3

        ns_scores.append(round(score, 1))
        ns_counted.append(qualifies)
        z_dpbs.append(round(z_dpb, 3))
        z_srs.append(round(z_sr, 3))
        z_spbs.append(round(z_spb, 3))

    result = df.copy()
    result["newscore_v3"]         = ns_scores
    result["newscore_v3_counted"] = ns_counted  # False = below min-battles, not in overall
    result["ns_v3_z_dpb"] = z_dpbs
    result["ns_v3_z_sr"]  = z_srs
    result["ns_v3_z_spb"] = z_spbs
    return result


def calc_overall_newscore_v3(df_ns: pd.DataFrame) -> float:
    """Battle-weighted average NewScore v3 across qualifying tanks (counted only)."""
    if "newscore_v3_counted" in df_ns.columns:
        valid = df_ns[df_ns["newscore_v3_counted"] == True].dropna(subset=["newscore_v3"])
    else:
        valid = df_ns.dropna(subset=["newscore_v3"])
    if valid.empty:
        return 0.0
    return float(
        (valid["newscore_v3"] * valid["battles"]).sum() / valid["battles"].sum()
    )


# ── NewScore v4 — canonical ───────────────────────────────────────────────────
# Constants are the output of scripts/v4_research.py (research/v4_constants.json);
# test_newscore.py fails if the two drift apart. See research/v4_report.md.
W_DPB_V4 = 0.394
W_SR_V4  = 0.355
W_SPB_V4 = 0.251

MU_V4 = -0.0918     # composite of the median account        -> 500
S_V4  = 0.9632      # set so the 99th-percentile account     -> 900

N0_V4     = 13.5    # battles at which a tank's own results and the account average weigh equally
SIGMA2_V4 = 5.13    # per-battle sampling variance of the composite (approximate)

V4_REQUIRED_COLUMNS = ("tank_id", "battles", "damage_dealt", "survived_battles", "spotted")
V4_OUTPUT_COLUMNS = ("newscore_v4", "newscore_v4_raw", "newscore_v4_counted",
                     "ns_v4_composite", "ns_v4_impact",
                     "ns_v4_z_dpb", "ns_v4_z_sr", "ns_v4_z_spb")


def scale_v4(composite: float) -> float:
    """
    Map a v4 composite to 0–1000. Logistic: exactly 500 at MU_V4, strictly
    increasing and smooth everywhere, approaching 0 and 1000 without a clamp.
    Steepest at 500, where one composite unit is worth 250 / S_V4 points.
    """
    return 1000.0 / (1.0 + math.exp(-(composite - MU_V4) / S_V4))


def unscale_v4(score: float) -> float:
    """Inverse of scale_v4, for a score strictly between 0 and 1000."""
    return MU_V4 + S_V4 * math.log(score / (1000.0 - score))


def _scale_v4_np(composite: np.ndarray) -> np.ndarray:
    return 1000.0 / (1.0 + np.exp(-(composite - MU_V4) / S_V4))


_norm_ev_v4: dict | None = None


def _load_norm_ev_v4(path: str | Path | None = None) -> dict:
    """
    Load per-tank normalization EVs for v4 (dpb, sr, spb) from tankavg.json.

    Nothing is defaulted. A tank is skipped unless it has a meanSd block,
    number_of_players >= MIN_TANK_POPULATION, and a finite mean plus a finite
    positive SD for all three stats — the older loaders fall back to SD=1,
    which would turn a missing SD into a huge z-score. Skipped tanks flow
    through the `tid not in norm_ev -> NaN` path of the scorer.

    Pass `path` to read another tankavg.json-format file; that result is not
    cached.
    """
    global _norm_ev_v4
    if path is None and _norm_ev_v4 is not None:
        return _norm_ev_v4
    with open(path or Path(__file__).parent / "tankavg.json", encoding="utf-8") as f:
        raw = json.load(f)
    keys = {
        "dpbMean": "dpbMean", "dpbSd": "dpbSd",
        "srMean": "survivalRateMean", "srSd": "survivalRateSd",
        "spbMean": "spbMean", "spbSd": "spbSd",
    }
    norm_ev = {}
    for entry in raw:
        tid = entry.get("tank_id")
        ms = entry.get("meanSd")
        if not tid or not ms:
            continue
        if entry.get("number_of_players", 0) < MIN_TANK_POPULATION:
            continue
        ev = {k: ms.get(src) for k, src in keys.items()}
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in ev.values()):
            continue
        if min(ev["dpbSd"], ev["srSd"], ev["spbSd"]) <= 0:
            continue
        norm_ev[int(tid)] = {k: float(v) for k, v in ev.items()}
    if path is None:
        _norm_ev_v4 = norm_ev
    return norm_ev


def calc_newscore_v4_per_tank(df: pd.DataFrame, norm_ev: dict | None = None) -> pd.DataFrame:
    """
    Add the NewScore v4 columns to a copy of df, which must hold ONE account's
    tanks (the per-tank display score and the impact column depend on the
    account's own composite; rows of several accounts would be scored as one).

    Requires df columns: tank_id, battles, damage_dealt, survived_battles,
    spotted — a missing column raises instead of being read as zero.
    norm_ev defaults to _load_norm_ev_v4() (tankavg.json).

    A tank is counted when it has norms, battles > 0 and finite stats; there
    is no minimum-battle threshold. Columns added:
        ns_v4_z_dpb / _z_sr / _z_spb   z-scores against the tank's pool, capped at +/-Z_CAP
        ns_v4_composite                the weighted z-scores (what the account score averages)
        newscore_v4_raw                the tank's own composite on the 0–1000 scale
        newscore_v4                    display score: the composite shrunk toward the
                                       account's composite by battles / (battles + N0_V4),
                                       so a handful of battles cannot produce an extreme
        ns_v4_impact                   points this tank adds to (+) or takes off (-) the
                                       account score: score with it minus score without it
                                       (NaN for the only counted tank)
        newscore_v4_counted            False, with NaN in every column above, for a tank
                                       without norms, without battles, or with a stat
                                       that is not a finite number
    """
    if df.empty:
        # an account with no tanks (build_tank_dataframe returns a column-less frame)
        result = df.copy()
        for col in V4_OUTPUT_COLUMNS:
            result[col] = pd.Series(dtype=bool if col == "newscore_v4_counted" else float)
        return result
    missing = [c for c in V4_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"calc_newscore_v4_per_tank: missing column(s) {missing}")
    if norm_ev is None:
        norm_ev = _load_norm_ev_v4()

    def num(col: str) -> np.ndarray:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    battles = num("battles")
    evs = [norm_ev.get(int(tid)) if pd.notna(tid) else None for tid in df["tank_id"]]

    def zscore(per_battle: np.ndarray, mean_key: str, sd_key: str) -> np.ndarray:
        mean = np.array([ev[mean_key] if ev else np.nan for ev in evs])
        sd   = np.array([ev[sd_key]   if ev else np.nan for ev in evs])
        return np.clip((per_battle - mean) / sd, -Z_CAP, Z_CAP)

    damage, survived, spotted = num("damage_dealt"), num("survived_battles"), num("spotted")
    with np.errstate(divide="ignore", invalid="ignore"):
        z_dpb = zscore(damage / battles,           "dpbMean", "dpbSd")
        z_sr  = zscore(survived / battles * 100.0, "srMean",  "srSd")
        z_spb = zscore(spotted / battles,          "spbMean", "spbSd")
        composite = W_DPB_V4 * z_dpb + W_SR_V4 * z_sr + W_SPB_V4 * z_spb

    # a tank counts only if everything about it is a real number; the rest are blanked
    # (the z-cap would otherwise turn an infinite stat into a tidy +5)
    counted = ((battles > 0) & np.isfinite(battles) & np.isfinite(damage) & np.isfinite(survived)
               & np.isfinite(spotted) & np.isfinite(composite))
    z_dpb, z_sr, z_spb, composite = (np.where(counted, a, np.nan) for a in (z_dpb, z_sr, z_spb, composite))
    b = np.where(counted, battles, 0.0)
    comp0 = np.where(counted, composite, 0.0)

    total = b.sum()
    if total > 0:
        acct = (b * comp0).sum() / total
        shrunk = (b * composite + N0_V4 * acct) / (b + N0_V4)       # NaN where not counted
        rest = total - b
        with np.errstate(divide="ignore", invalid="ignore"):
            without = np.where(counted & (rest > 0), (total * acct - b * comp0) / rest, np.nan)
        impact = scale_v4(acct) - _scale_v4_np(without)
    else:
        shrunk = impact = np.full(len(df), np.nan)

    result = df.copy()
    result["newscore_v4"]         = np.round(_scale_v4_np(shrunk), 1)
    result["newscore_v4_raw"]     = np.round(_scale_v4_np(composite), 1)
    result["newscore_v4_counted"] = counted
    result["ns_v4_composite"]     = composite
    result["ns_v4_impact"]        = np.round(impact, 2)
    result["ns_v4_z_dpb"] = np.round(z_dpb, 3)
    result["ns_v4_z_sr"]  = np.round(z_sr, 3)
    result["ns_v4_z_spb"] = np.round(z_spb, 3)
    return result


def newscore_v4_summary(df_ns: pd.DataFrame) -> dict | None:
    """
    Account-level v4 figures from the output of calc_newscore_v4_per_tank, or
    None when no tank is counted:
        score      the account score (0–1000)
        composite  battle-weighted mean composite of the counted tanks
        battles    counted battles
        se         roughly one standard error of the score from battle-to-battle
                   luck alone (it shrinks with sqrt(battles)). SIGMA2_V4 is only
                   pinned down to about a quarter, and this says nothing about
                   how the score would move with a different choice of tanks —
                   that is the larger uncertainty (research/v4_report.md, 3.7)
        z_dpb, z_sr, z_spb   battle-weighted means of the three z-scores
    """
    valid = df_ns[df_ns["newscore_v4_counted"] == True]
    if valid.empty:
        return None
    b = valid["battles"].to_numpy(dtype=float)
    total = b.sum()

    def mean(col: str) -> float:
        return float((valid[col].to_numpy(dtype=float) * b).sum() / total)

    composite = mean("ns_v4_composite")
    score = scale_v4(composite)
    slope = score * (1000.0 - score) / (1000.0 * S_V4)        # points per composite unit here
    return {
        "score": score,
        "composite": composite,
        "battles": float(total),
        "se": slope * math.sqrt(SIGMA2_V4 / total),
        "z_dpb": mean("ns_v4_z_dpb"),
        "z_sr": mean("ns_v4_z_sr"),
        "z_spb": mean("ns_v4_z_spb"),
    }


def calc_overall_newscore_v4(df_ns: pd.DataFrame) -> float:
    """
    NewScore v4 account score: the battle-weighted mean composite of every
    counted tank, mapped to 0–1000 once. (Not the mean of the per-tank
    scores — the scale is curved, and averaging after it is what made v0–v3
    depend on how performance was spread across tanks.)

    Returns 0.0 when no tank is counted, as the older versions do; a scored
    account can never be exactly 0, and newscore_v4_summary returns None in
    that case, which is the cleaner test.
    """
    summary = newscore_v4_summary(df_ns)
    return summary["score"] if summary else 0.0
