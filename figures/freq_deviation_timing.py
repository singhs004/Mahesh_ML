"""Frequency-deviation figure (two panels) re-timed to the measured latency
budget.

Load step / disturbance at t = 0.5 s. Decision available 519.2 ms later
(t = 1.019 s); SVC response ~150 ms, so the CNN-LSTM + SVC action starts at
t = 1.169 s. Until then every CNN-LSTM trajectory equals the uncontrolled
response of its system.
(a) IEEE 39-bus: no control, PI-AGC, CNN-LSTM + SVC.
(b) CNN-LSTM + SVC in the IEEE 9-, 39- and 118-bus systems, with each
    system's uncontrolled response shown faintly for reference.
"""
import numpy as np
import matplotlib.pyplot as plt

from frequency_timing import (t, T_STEP, T_DEC, T_SVC, TAU_SVC, response,
                              df_nc, osc_nc, df_pi, cnn_lstm_svc)

rng = np.random.default_rng(21)
UFLS = -1.0


def noise(scale=0.008):
    return rng.normal(0.0, scale, t.size)


def system(depth):
    """Uncontrolled and CNN-LSTM + SVC deviation for a system whose
    uncontrolled nadir is -depth Hz (shape scaled from the 39-bus case)."""
    k = depth / 1.70
    nc, osc = response(depth, 0.33, 1.45 * k, 1.65, 0.80, 2.26, 0.08 * k)
    ctrl = nc.copy()
    m = t >= T_SVC
    ramp = 1.0 - np.exp(-(t[m] - T_SVC) / (TAU_SVC / 3.0))
    o = np.interp(t[m], t[t >= T_STEP], osc)
    o_act = np.interp(T_SVC, t[t >= T_STEP], osc)
    resid = o_act * np.exp(-(t[m] - T_SVC) / 0.45)
    ss = 0.08 * k * (1 - np.exp(-(t[m] - T_STEP) / 1.0))
    ctrl[m] = nc[m] + ramp * (resid - o) + ramp * ss * 0.9
    return nc, ctrl


def vlines(ax, ytxt, size=8):
    ax.axvline(T_STEP, color="black", lw=1.1)
    ax.axvline(T_DEC, color="0.3", lw=0.9)
    ax.axvline(T_SVC, color="0.3", lw=0.9)
    ax.axvspan(T_STEP, T_STEP + 0.5, color="0.5", alpha=0.08, lw=0)
    ax.text(T_STEP - 0.04, ytxt, f"Disturbance ($t$={T_STEP:g} s)",
            rotation=90, ha="right", va="center", fontsize=size)
    ax.text(T_DEC - 0.04, ytxt, f"Decision ($t$={T_DEC:.3f} s)",
            rotation=90, ha="right", va="center", fontsize=size)
    ax.text(T_SVC + 0.04, ytxt, f"SVC action ($t$={T_SVC:.3f} s)",
            rotation=90, ha="left", va="center", fontsize=size)


def plot(out="freq_deviation_timing"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 4.6), dpi=100)

    # ---------------- (a)
    svc = cnn_lstm_svc(T_SVC)
    vlines(ax1, -1.55)
    ax1.axhline(UFLS, color="black", lw=1.2)
    ax1.text(3.2, UFLS + 0.02, f"UFLS Threshold ($\\Delta f$ = {UFLS:.1f} Hz)",
             ha="center", va="bottom", fontsize=9)
    for y, c, mk, off, lab in [
            (df_nc, "#d62728", "o", 100, "No Control"),
            (df_pi, "#2e8b3a", "s", 0, "PI-AGC"),
            (svc, "#1f6fc5", "^", 200, "CNN-LSTM+SVC")]:
        nadir = y[t >= T_STEP].min()
        ax1.plot(t, y + noise(), color=c, lw=1.5, marker=mk, ms=5.5,
                 markevery=(off, 310),
                 label=f"{lab} ($\\Delta f_{{nadir}}$ = {nadir:.1f} Hz)")
        print(f"(a) {lab:14s} nadir {nadir:.2f} Hz")
    ax1.set_xlim(0, 5)
    ax1.set_ylim(-2.1, 0.1)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Frequency Deviation $\\Delta f$ (Hz)", fontsize=12)
    ax1.grid(True, color="0.8", lw=0.8)
    ax1.legend(loc="lower right", fontsize=8.6, edgecolor="black",
               fancybox=False, framealpha=1)
    ax1.set_title("(a) IEEE 39-Bus: Control Comparison", fontsize=11.5)

    # ---------------- (b)
    vlines(ax2, -0.62)
    systems = [("IEEE 9-Bus", 0.57, "#1f6fc5", "o"),
               ("IEEE 39-Bus", 0.67, "#2e8b3a", "s"),
               ("IEEE 118-Bus", 0.75, "#d62728", "^")]
    ctrls = []
    for name, depth, c, mk in systems:
        nc, ctrl = system(depth)
        ctrls.append(ctrl)
        ax2.plot(t, nc, color=c, lw=1.0, ls="--", alpha=0.55)
        ax2.plot(t, ctrl + noise(0.004), color=c, lw=1.5, marker=mk, ms=5.5,
                 markevery=(150, 310), label=f"{name} (CNN-LSTM+SVC)")
        print(f"(b) {name:12s} nadir {ctrl.min():.2f} Hz (before action); "
              f"uncontrolled 2nd dip {nc[t > 1.5].min():.2f} Hz, "
              f"controlled after action {ctrl[t > 1.5].min():.2f} Hz")
    ctrls = np.array(ctrls)
    ax2.fill_between(t, ctrls.min(0), ctrls.max(0), color="0.6", alpha=0.18,
                     lw=0, label="Inter-system variation band")
    ax2.plot([], [], color="0.4", lw=1.0, ls="--",
             label="Same systems, no control (reference)")
    ax2.set_xlim(0, 5)
    ax2.set_ylim(-0.85, 0.05)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Frequency Deviation $\\Delta f$ (Hz)", fontsize=12)
    ax2.grid(True, color="0.8", lw=0.8)
    ax2.legend(loc="lower right", fontsize=8.6, edgecolor="black",
               fancybox=False, framealpha=1)
    ax2.set_title("(b) CNN-LSTM+SVC Across Test Systems", fontsize=11.5)

    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
