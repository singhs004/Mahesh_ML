"""System frequency after the G6 trip (shared model freq_model.py)."""
import matplotlib.pyplot as plt

import freq_model as fm
import freq_plotlib as fp
import style


def plot(out="frequency_timing"):
    style.apply()
    fig, ax = plt.subplots(figsize=(7.6, 4.9), dpi=100)
    fp.event_lines(ax, ytxt=48.25)
    ax.axhline(49.0, color="black", lw=1.2)
    ax.text(7.5, 49.02, "UFLS stage-1 threshold (49.0 Hz)", ha="center",
            va="bottom", fontsize=9)
    for key, case, tc, c, ls, lw, lab in fp.CASES:
        r = fp.run(key, case, tc)
        mt = fm.metrics(r)
        ax.plot(r["t"], r["f"], color=c, ls=ls, lw=lw,
                label=f"{lab} (nadir {mt['nadir']:.2f} Hz)")
    ax.set_xlim(0, 10)
    ax.set_ylim(47.8, 50.2)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("System Frequency (Hz)", fontsize=12)
    ax.grid(True, color="0.8", lw=0.8)
    ax.legend(loc="lower right", fontsize=8.6, edgecolor="black",
              fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
