import math

try:
    import streamlit as st
except ImportError:  # the HTML export only needs the colour helpers
    st = None

WN8_BANDS = [
    (0,    300,  "#930D0D", "Very Bad"),
    (300,  450,  "#CD3333", "Bad"),
    (450,  650,  "#CC7A00", "Below Average"),
    (650,  900,  "#CCB800", "Average"),
    (900,  1200, "#849B24", "Above Average"),
    (1200, 1600, "#4D7326", "Good"),
    (1600, 2000, "#4099BF", "Very Good"),
    (2000, 2450, "#3972C6", "Great"),
    (2450, 2900, "#A83BB5", "Unicum"),
    (2900, 99999, "#793DB6", "Super Unicum"),
]


def wn8_color(wn8: float) -> str:
    if wn8 is None or math.isnan(wn8):
        return "#888888"
    for lo, hi, color, _ in WN8_BANDS:
        if lo <= wn8 < hi:
            return color
    return WN8_BANDS[-1][2]


def wn8_label(wn8: float) -> str:
    if wn8 is None or math.isnan(wn8):
        return "N/A"
    for lo, hi, _, label in WN8_BANDS:
        if lo <= wn8 < hi:
            return label
    return WN8_BANDS[-1][3]


def render_wn8_badge(wn8: float, subtitle: str = "") -> None:
    if st is None:
        raise RuntimeError("render_wn8_badge needs Streamlit")
    color = wn8_color(wn8)
    label = wn8_label(wn8)
    wn8_str = f"{wn8:,.0f}" if not math.isnan(wn8) else "N/A"
    html = f"""
    <div style="
        background-color: {color};
        border-radius: 12px;
        padding: 20px 32px;
        display: inline-block;
        text-align: center;
        min-width: 180px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    ">
        <div style="color: white; font-size: 48px; font-weight: 800; line-height: 1;">
            {wn8_str}
        </div>
        <div style="color: rgba(255,255,255,0.9); font-size: 18px; font-weight: 600; margin-top: 4px;">
            {label}
        </div>
        {"<div style='color: rgba(255,255,255,0.7); font-size: 13px; margin-top: 2px;'>" + subtitle + "</div>" if subtitle else ""}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def color_wn8_cell(val) -> str:
    """CSS background color for a WN8 value in a pandas Styler."""
    try:
        v = float(val)
        if math.isnan(v):
            return ""
        color = wn8_color(v)
        return f"background-color: {color}; color: white; font-weight: bold;"
    except (TypeError, ValueError):
        return ""


def color_pct_cell(val) -> str:
    """
    CSS for % of expected columns.
    < 80%  → red
    80-99% → amber
    >= 100% → green
    """
    try:
        v = float(val)
        if math.isnan(v):
            return ""
        if v >= 100:
            return "background-color: #2d6a2d; color: white;"
        elif v >= 80:
            return "background-color: #a67c00; color: white;"
        else:
            return "background-color: #8b1a1a; color: white;"
    except (TypeError, ValueError):
        return ""


RATIO_BANDS = [
    (0,   130, "#8b1a1a"),
    (130, 140, "#CD3333"),
    (140, 148, "#CC7A00"),
    (148, 156, "#CCB800"),
    (156, 164, "#4D7326"),
    (164, 172, "#4099BF"),
    (172, 180, "#3972C6"),
    (180, 999, "#793DB6"),
]


def color_ratio_cell(val) -> str:
    """
    CSS for the damage-ratio column (your dmg/battle as % of the tank's
    expected damage). Bands are centred on ~152%, the account-wide average,
    so the colour shows how a tank compares to your own typical output
    rather than to the population.
    """
    try:
        v = float(val)
        if math.isnan(v):
            return ""
        for lo, hi, color in RATIO_BANDS:
            if lo <= v < hi:
                return f"background-color: {color}; color: white; font-weight: bold;"
        return ""
    except (TypeError, ValueError):
        return ""


def color_drag_cell(val) -> str:
    """Green for negative drag (above avg), red for positive drag (below avg)."""
    try:
        v = float(val)
        if v > 0:
            return "color: #ff6b6b;"
        elif v < 0:
            return "color: #69db7c;"
        return ""
    except (TypeError, ValueError):
        return ""
