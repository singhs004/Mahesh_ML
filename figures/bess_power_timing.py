"""Generator and BESS active power after the G6 trip (CNN-LSTM + BESS case),
from the shared model freq_model.py. Generator set-points and headroom are
those of Table 4 of the paper."""
import numpy as np
import matplotlib.pyplot as plt

import freq_model as fm
import freq_plotlib as fp
import style


def plot(out="bess_power_timing"):
    style.apply()
    r = fp.run("PF", "PF", 0.0)
    t, g, b = r["t"], r["gen"], r["bess"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 7.8), dpi=100)
    for ax in (ax1, ax2):
        fp.event_lines(ax)
        ax.grid(True, color="0.85", lw=0.8)
        ax.set_xlim(-0.3, 10.3)

    for name, c, ls, role in [("G1", "#1f6fb4", "-", "governor"),
                              ("G3", "#228b3b", "--", "governor + AGC"),
                              ("G9", "#8b6508", "-.", "governor + AGC"),
                              ("G10", "#7d3c98", (0, (5, 1, 1, 1)),
                               "governor + AGC")]:
        rat, p0, head = fm.GEN[name]
        y = g[name]
        ax1.plot(t, y, color=c, ls=ls, lw=2.2,
                 label=f"{name} — {role} ({p0} MW, headroom {head} MW; "
                       f"+{y[-1] - p0:.0f} MW at 10 s)")
    ax1.plot(t, g["G6"], color="#d62728", ls=":", lw=2.4,
             label=f"G6 — trip at $t$={fm.T_EVT:.2f} s (−{fm.P_TRIP:.0f} MW)")
    ax1.set_ylim(-30, 1250)
    ax1.set_ylabel("Active power $P$ (MW)", fontsize=12)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_title("(a) Synchronous Generator Active Power (Table 4 set-points)",
                  fontsize=11.5)
    ax1.legend(loc="lower right", bbox_to_anchor=(0.995, 0.05), fontsize=7.8,
               edgecolor="black", fancybox=False, framealpha=1)

    tot = b["Bus 3"] + b["Bus 29"]
    i_full = np.argmax(tot >= 127.9)
    fs = (t >= 0.5) & (t <= 0.7)
    ax2.axvspan(0.5, 0.7, color="#f5b041", alpha=0.35, lw=0)
    ax2.fill_between(t, 0, b["Bus 29"], color="#8e44ad", alpha=0.15, lw=0)
    ax2.fill_between(t, 0, b["Bus 3"], color="#1e9e55", alpha=0.15, lw=0)
    ax2.plot(t, b["Bus 3"], color="#1e9e55", lw=2.6,
             label=f"BESS at Bus 3 (+50 MW, ramp {fm.BESS_RAMP:.0f} MW/s)")
    ax2.plot(t, b["Bus 29"], color="#8e44ad", lw=2.6, ls="--",
             label=f"BESS at Bus 29 (+78 MW, ramp {fm.BESS_RAMP:.0f} MW/s)")
    ax2.annotate(f"First-swing window (0.50–0.70 s):\n"
                 f"{tot[fs].max():.0f} MW delivered", xy=(0.6, 100),
                 xytext=(2.0, 100), color="#b9770e", fontsize=9.5,
                 va="center",
                 arrowprops=dict(arrowstyle="->", color="#b9770e", lw=0.9))
    ax2.annotate(f"128 MW reached at $t$={t[i_full]:.3f} s",
                 xy=(t[i_full], 78), xytext=(2.3, 64), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=0.9))
    ax2.text(fm.T_DEC - 0.06, 4, f"CNN-LSTM dispatch ($t$={fm.T_DEC:.3f} s)",
             rotation=90, ha="right", va="bottom", fontsize=8.3,
             bbox=dict(fc="white", ec="none", pad=0.2))
    ax2.set_ylim(0, 115)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("BESS active power (MW)", fontsize=12)
    ax2.set_title("(b) Battery Energy Storage System (BESS) Active-Power "
                  "Response", fontsize=11.5)
    ax2.legend(loc="upper right", fontsize=8.6, edgecolor="black",
               fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
