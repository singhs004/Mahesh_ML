"""G10 communication-delay sensitivity (CNN-LSTM + STATCOM) from the shared
model in g10_model.py."""
import numpy as np
import matplotlib.pyplot as plt

import g10_model as m
import style

T_COMMS = [0.0, 0.100, 0.200, 0.300, 0.400, 0.500]


def plot(out="delay_sensitivity_timing"):
    style.apply()
    crit = m.critical_tcomm("STATCOM")
    colors = ["#27a050", "#5dade2", "#f39c12", "#e67e22", "#e74c3c", "#922b21"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=100)
    ax.axhspan(180, 200, color="#e74c3c", alpha=0.08, lw=0)
    style.event_lines(ax, m, labels=True, y=130, devices=("STATCOM",))
    nc, _ = m.g10("T")
    ax.plot(m.t, nc, color="0.55", lw=1.6, ls=(0, (1, 1.5)),
            label="No control (unstable)")
    for tc, c in zip(T_COMMS, colors):
        y, _ = m.g10("T", "STATCOM", tc)
        stable = y.max() < 180
        st = (f"$\\delta_{{peak}}$ = {y.max():.1f}°" if stable
              else "loss of synchronism")
        ax.plot(m.t, y, color=c, lw=2.4, ls="-" if stable else "--",
                label=f"$T_{{comm}}$ = {tc*1e3:.0f} ms, act. "
                      f"{m.T_ACT['STATCOM'] + tc:.3f} s ({st})")
        if stable:
            i = np.argmax(y)
            ax.plot(m.t[i], y[i], "o", color=c, ms=5, zorder=4)
    ax.axhline(180, color="black", ls="--", lw=1.3,
               label="180° stability boundary")
    ax.text(m.T_END - 0.04, 189,
            f"Critical $T_{{comm}}$ ≈ {crit*1e3:.0f} ms (STATCOM action at "
            f"{m.T_ACT['STATCOM'] + crit:.3f} s)", ha="right", va="center",
            fontsize=9, color="#922b21",
            bbox=dict(fc="white", ec="none", pad=0.4))
    ax.set_xlim(0, m.T_END)
    ax.set_ylim(0, 200)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("G10 rotor angle $\\delta$ (degrees)", fontsize=12)
    ax.grid(True, color="0.85", lw=0.8)
    ax.legend(loc="center right", bbox_to_anchor=(0.995, 0.47), fontsize=8.0,
              edgecolor="black", fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
