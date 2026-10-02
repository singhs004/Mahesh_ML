"""G10 rotor-angle delay sensitivity re-timed to the measured latency budget.

Each curve is the CNN-LSTM + STATCOM response for a different communication
delay T_comm. The corrective action starts at
    t_act = t_onset + 500 ms window + 1.9 ms + 17.3 ms + 30 ms + T_comm
          = 0.649 s + T_comm
and the G10 model is the same one used in rotor_angle_timing.py, so the two
figures are consistent.
"""
import numpy as np
import matplotlib.pyplot as plt

from rotor_angle_timing import (t, T_FAULT, T_CLEAR, T_DEC, T_STATCOM,
                                TAU_STATCOM, g10_nc, g10_controlled)

T_COMMS = [0.0, 0.100, 0.200, 0.300, 0.350, 0.500]
LIMIT = 180.0


def critical_tcomm(lo=0.0, hi=0.5, tol=1e-3):
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if g10_controlled(T_STATCOM + mid, TAU_STATCOM).max() < LIMIT:
            lo = mid
        else:
            hi = mid
    return lo


def plot(out="delay_sensitivity_timing"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    })
    colors = ["#27a050", "#5dade2", "#f39c12", "#e67e22", "#e74c3c", "#922b21"]
    tc_crit = critical_tcomm()
    print(f"critical T_comm = {tc_crit*1e3:.0f} ms "
          f"(action at {T_STATCOM + tc_crit:.3f} s)")

    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=100)
    ax.axhspan(LIMIT, 200, color="#e74c3c", alpha=0.08, lw=0)
    ax.axvspan(T_FAULT, T_FAULT + 0.5, color="0.5", alpha=0.08, lw=0)
    ax.plot(t, g10_nc, color="0.55", lw=1.6, ls=(0, (1, 1.5)),
            label="No control (unstable)")

    for tc, c in zip(T_COMMS, colors):
        t_act = T_STATCOM + tc
        y = g10_controlled(t_act, TAU_STATCOM)
        peak = y.max()
        stable = peak < LIMIT
        status = f"$\\delta_{{peak}}$ = {peak:.1f}°" if stable \
            else "loss of synchronism"
        lab = (f"$T_{{comm}}$ = {tc*1e3:.0f} ms, act. {t_act:.3f} s "
               f"({status})")
        ax.plot(t, y, color=c, lw=2.4, ls="-" if stable else "--", label=lab)
        print(lab)
        if stable:
            i = np.argmax(y)
            ax.plot(t[i], peak, "o", color=c, ms=5, zorder=4)

    ax.axhline(LIMIT, color="black", ls="--", lw=1.3,
               label="180° stability boundary")
    for x, txt, side in [(T_FAULT, f"Fault onset ($t$={T_FAULT:g} s)", "l"),
                         (T_CLEAR, f"Fault cleared ($t$={T_CLEAR:g} s)", "l"),
                         (T_DEC, f"Decision ($t$={T_DEC:.3f} s)", "l"),
                         (T_STATCOM, f"Earliest STATCOM action "
                          f"($t$={T_STATCOM:.3f} s)", "r")]:
        ax.axvline(x, color="black", lw=0.9, ls="--" if x == T_CLEAR else "-")
        dx = -0.025 if side == "l" else 0.025
        ax.text(x + dx, 125, txt, rotation=90, va="center", fontsize=8.5,
                ha="right" if side == "l" else "left")

    ax.text(2.96, 189, f"Critical $T_{{comm}}$ ≈ {tc_crit*1e3:.0f} ms "
            f"(total ≈ {(T_STATCOM + tc_crit - T_FAULT)*1e3:.0f} ms "
            "after onset)", ha="right", va="bottom", fontsize=9,
            color="#922b21", bbox=dict(fc="white", ec="none", pad=0.6))

    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 200)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("G10 rotor angle $\\delta$ (degrees)", fontsize=12)
    ax.grid(True, color="0.85", lw=0.8)
    ax.legend(loc="center right", bbox_to_anchor=(0.995, 0.50), fontsize=8.2,
              edgecolor="black", fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
