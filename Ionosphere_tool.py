# ionosphere_tool.py
# Refraction in the Ionosphere — NEETS Mod 10 §2.6
# Single-view path. Stratified index + Snell's law.
# Starts at TX tip; if the hop reaches RX, the ray ends at RX tip.

import math
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
import streamlit as st

# -------------------- Physics (NEETS-level) --------------------

def critical_frequency_mhz(N12: float) -> float:
    """fo (MHz) from peak electron density N in 1e12 e-/m^3: fo ≈ 9√N."""
    N12 = max(0.0, float(N12))
    return 9.0 * math.sqrt(N12)

def muf_from_fo_alpha(fo_mhz: float, takeoff_deg: float) -> float:
    """MUF = fo / sin(alpha), alpha measured above the horizon."""
    s = math.sin(math.radians(max(1e-6, min(89.999, takeoff_deg))))
    return fo_mhz / s

def plasma_freq_profile_mhz(y_km: np.ndarray, fo_mhz: float, H_km: float, sigma_km: float = 70.0) -> np.ndarray:
    """Gaussian layer centered at H with peak fo."""
    return fo_mhz * np.exp(-0.5 * ((y_km - H_km) / sigma_km) ** 2)

def refractive_index(y_km: np.ndarray, f_mhz: float, fo_mhz: float, H_km: float, sigma_km: float = 70.0) -> np.ndarray:
    """n(y) = sqrt(1 - (fp/f)^2)."""
    fp = plasma_freq_profile_mhz(y_km, fo_mhz, H_km, sigma_km)
    val = 1.0 - (fp / max(1e-9, f_mhz)) ** 2
    return np.sqrt(np.clip(val, 1e-6, 1.0))

def integrate_ray(alpha0_deg: float, f_mhz: float, fo_mhz: float, H_km: float,
                  y_top: float, y0: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Integrate the ray in a stratified ionosphere starting at height y0 (TX tip).
    Returns (x, y, R_firsthop (horizontal), returned?)
    - x and y start at (0, y0). Mirror descent to y0 if returned.
    """
    alpha0 = math.radians(alpha0_deg)
    C = math.cos(alpha0)  # Snell constant: n(y) * sin(phi) = cos(alpha0)

    dy = 1.0
    y = np.arange(y0, y_top + dy, dy)
    n = refractive_index(y, f_mhz, fo_mhz, H_km)

    # Turning condition: fp >= f * sin(alpha0)
    target_fp = f_mhz * math.sin(alpha0)
    fp = plasma_freq_profile_mhz(y, fo_mhz, H_km)
    returned = (fp.max() + 1e-9) >= target_fp

    if returned:
        idx = np.argmax(fp >= target_fp)
        if idx == 0:
            y_turn = y[0]
        else:
            y1, y2 = y[idx - 1], y[idx]
            fp1, fp2 = fp[idx - 1], fp[idx]
            y_turn = y1 + (target_fp - fp1) * (y2 - y1) / max(1e-9, (fp2 - fp1))

        y_up = np.arange(y0, y_turn + dy, dy)
        n_up = refractive_index(y_up, f_mhz, fo_mhz, H_km)
        cos_a = np.clip(C / n_up, 0.0, 0.999999)
        sin_a = np.sqrt(1.0 - cos_a ** 2)
        slope = cos_a / np.clip(sin_a, 1e-9, None)   # cot(alpha)
        x_up = np.cumsum(slope) * dy
        x_up = np.insert(x_up, 0, 0.0)[:len(y_up)]

        # symmetric descent back to y0
        x_turn = x_up[-1]
        y_down = y_up[::-1]
        x_down = x_turn + (x_turn - x_up[::-1])

        x = np.concatenate([x_up, x_down[1:]])
        y_all = np.concatenate([y_up, y_down[1:]])
        R = 2.0 * x_turn
        return x, y_all, R, True
    else:
        cos_a = np.clip(C / n, 0.0, 0.999999)
        sin_a = np.sqrt(1.0 - cos_a ** 2)
        slope = cos_a / np.clip(sin_a, 1e-9, None)
        x = np.cumsum(slope) * dy
        x = np.insert(x, 0, 0.0)[:len(y)]
        return x, y, float("nan"), False

def fading_line(ax, x, y, color="#f59e0b", a0=1.0, a1=0.35, lw=4.0):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    rgba = np.tile(to_rgba(color), (len(segs), 1))
    rgba[:, 3] = np.linspace(a0, a1, len(segs))
    lc = LineCollection(segs, colors=rgba, linewidths=lw, capstyle="round")
    ax.add_collection(lc)

# -------------------- Scene / styling --------------------

COLORS = {
    "sky_top":   "#78ba8f",   # greenish top
    "sky_bot":   "#0f3a6b",   # deep blue base
    "ground":    "#0b2746",
    "iono_lo":   "#8fb7d6",
    "iono_mid":  "#6ea0cb",
    "iono_hi":   "#9dc2df",
    "ray":       "#f59e0b",
    "rx":        "#e11d48",
    "tx":        "#16a34a",
    "ok":        "#10b981",
    "warn":      "#f59e0b",
    "err":       "#ef4444",
}

def gradient_background(ax, x_max, y_min, y_max):
    ax.add_patch(patches.Rectangle((0, (y_min+y_max)/2),
                                   x_max, (y_max-y_min)/2,
                                   facecolor=COLORS["sky_top"], alpha=0.45, lw=0))
    ax.add_patch(patches.Rectangle((0, y_min),
                                   x_max, (y_max-y_min)/2,
                                   facecolor=COLORS["sky_bot"], alpha=0.70, lw=0))

def draw_ionosphere(ax, x_max, H_km):
    thick = 220.0
    lo = H_km - 0.50*thick
    mid_lo = H_km - 0.22*thick
    mid_hi = H_km + 0.22*thick
    hi = H_km + 0.50*thick
    ax.fill_between([0, x_max], [lo, lo], [mid_lo, mid_lo], color=COLORS["iono_lo"], alpha=0.28, lw=0)
    ax.fill_between([0, x_max], [mid_lo, mid_lo], [mid_hi, mid_hi], color=COLORS["iono_mid"], alpha=0.40, lw=0)
    ax.fill_between([0, x_max], [mid_hi, mid_hi], [hi, hi], color=COLORS["iono_hi"], alpha=0.28, lw=0)
    ax.hlines(H_km, 0, x_max, colors="#2d6aa6", linestyles="dashed", linewidth=1.2, alpha=0.9)
    ax.text(x_max*0.5, hi+22, "IONOSPHERE", ha="center", va="bottom", fontsize=11, color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.25", fc="#2d6aa6", ec="none", alpha=0.85))
    return lo, mid_lo, mid_hi, hi

def draw_sites(ax, rx_km) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Draw TX/RX triangles (larger for visibility) and return their tip coordinates (x,y).
    TX tip ~ (tx_tip_x, tx_tip_y), RX tip ~ (rx_tip_x, rx_tip_y).
    """
    # Bigger, clearer TX
    tx_w, tx_h = 24.0, 40.0
    tx_tip_x, tx_tip_y = tx_w/2, tx_h
    tx_poly = np.array([[0, 0], [tx_w, 0], [tx_tip_x, tx_tip_y]])
    ax.add_patch(patches.Polygon(tx_poly, closed=True, fc=COLORS["tx"], ec="white", lw=1.4))
    ax.text(0, -12, "TX", ha="left", va="top", color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.15", fc=COLORS["tx"], ec="none", alpha=0.9), fontsize=9)

    # RX (same size, anchored to rx_km on ground)
    rx_w, rx_h = 24.0, 40.0
    rx_tip_x, rx_tip_y = rx_km - rx_w/2, rx_h
    rx_poly = np.array([[rx_km - rx_w, 0], [rx_km, 0], [rx_tip_x, rx_tip_y]])
    ax.add_patch(patches.Polygon(rx_poly, closed=True, fc=COLORS["rx"], ec="white", lw=1.4))
    ax.text(rx_km, -12, "RX", ha="right", va="top", color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.15", fc=COLORS["rx"], ec="none", alpha=0.9), fontsize=9)

    return (tx_tip_x, tx_tip_y), (rx_tip_x, rx_tip_y)

def badge(ax, x_max, y_max, text, color_hex):
    ax.text(x_max*0.99, y_max-10, text, ha="right", va="top", color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.25", fc=color_hex, ec="none", alpha=0.95), fontsize=10)

# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="Refraction in the Ionosphere — NEETS Mod 10", layout="wide")
st.title("Refraction in the Ionosphere — NEETS Mod 10")

left, mid, right = st.columns([1.4, 1.2, 1.4])
with left:
    N12 = st.slider("Ion Density (×10¹² e⁻/m³)", 0.05, 1.80, 0.64, 0.01)
    st.caption("Low ⟶ High")
with mid:
    takeoff = st.slider("Take-Off Angle α (°)", 5, 85, 60, 1)
    st.markdown(f"<div style='text-align:center'><b>{takeoff}°</b></div>", unsafe_allow_html=True)
with right:
    fopt = st.segmented_control("Frequency (MHz)", options=[3, 10, 25, "Custom"], default=25)
    if fopt == "Custom":
        f_mhz = float(st.number_input("Custom Frequency (MHz)", 0.5, 1000.0, 18.0, 0.5))
    else:
        f_mhz = float(fopt)

rx_km = st.slider("Receiver Distance (km)", 100, 3500, 1600, 50)
H_km  = st.slider("Layer Height H (km)", 80, 400, 250, 5)

start = st.button("Start", type="primary")
reset = st.button("Reset")
if reset:
    st.rerun()

# -------------------- Compute & draw --------------------

fo  = critical_frequency_mhz(N12)
muf = muf_from_fo_alpha(fo, takeoff)

y_top = max(2.0 * H_km, 480.0)

# First draw the scene to compute tip coordinates
xmax_for_layout = max(rx_km * 1.15, 2200)
ymin, ymax = -80, y_top
fig, ax = plt.subplots(figsize=(11.5, 5.4))
ax.set_xlim(0, xmax_for_layout)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Ground Range (km)")
ax.set_ylabel("Altitude (km)")
ax.grid(False)

gradient_background(ax, xmax_for_layout, ymin, ymax)
draw_ionosphere(ax, xmax_for_layout, H_km)
draw_earth(ax, xmax_for_layout)

(tx_tip_x, tx_tip_y), (rx_tip_x, rx_tip_y) = draw_sites(ax, rx_km)

# Integrate the ray starting at TX tip height
x_rel, y_vals, R_first, returned = integrate_ray(takeoff, f_mhz, fo, H_km, y_top, y0=tx_tip_y)

# Horizontal extent for final frame
xmax = max((np.nanmax(x_rel) + tx_tip_x) * 1.05, rx_km * 1.15, 2200)
ax.set_xlim(0, xmax)

# Repaint background to the new xmax
ax.patches.clear()
gradient_background(ax, xmax, ymin, ymax)
_, _, _, _ = draw_ionosphere(ax, xmax, H_km)
draw_earth(ax, xmax)
(tx_tip_x, tx_tip_y), (rx_tip_x, rx_tip_y) = draw_sites(ax, rx_km)

if start:
    # Offset the ray so it starts at the TX tip
    x_plot = x_rel + tx_tip_x
    y_plot = y_vals

    # If it returns, x_rel[-1] == R_first (back to y0). Compare against RX tip.
    if returned and not np.isnan(R_first):
        end_x_at_tip = tx_tip_x + R_first
        # If close, append a tiny straight segment to end exactly at RX tip
        tol = max(0.03 * rx_km, 30.0)
        if abs(end_x_at_tip - rx_tip_x) <= tol:
            # Make the last point exactly the RX tip for a crisp landing
            x_plot = np.append(x_plot, rx_tip_x)
            y_plot = np.append(y_plot, rx_tip_y)
            badge(ax, xmax, ymax, "Returned — Receiver Reached", COLORS["ok"])
            ax.plot([rx_km], [0], marker="o", ms=14, mec=COLORS["ok"], mfc="none", mew=3, alpha=0.9)
        elif end_x_at_tip < rx_tip_x:
            # Short of RX
            badge(ax, xmax, ymax, "Returned — Short of Receiver (skip distance)", COLORS["warn"])
        else:
            # Overshot RX
            badge(ax, xmax, ymax, "Returned — Overshot Receiver (skip distance)", COLORS["warn"])

    else:
        badge(ax, xmax, ymax, "Penetrated the Ionosphere (f > MUF)", COLORS["err"])

    # Draw the full path (single view) with gentle fading
    fading_line(ax, x_plot, y_plot, COLORS["ray"], a0=1.0, a1=0.35, lw=4.6)

st.pyplot(fig, use_container_width=True)

with st.expander("Notes"):
    st.write(
        "- **fo** increases with ion density: fo≈9√N (MHz).\n"
        "- **MUF** for take-off angle α: MUF = fo / sin α.\n"
        "- If f ≤ MUF the wave refracts and returns; else it penetrates.\n"
        "- Path bends in the dense part of the ionosphere; lower f or higher density ⇒ stronger bend (shorter hop)."
    )
