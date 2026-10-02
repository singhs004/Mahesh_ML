"""Frequency deviation (a) IEEE 39-bus control comparison and (b) CNN-LSTM +
BESS across test systems, from the shared model freq_model.py."""
import numpy as np
import matplotlib.pyplot as plt

import freq_model as fm
import freq_plotlib as fp
import style


def plot(out="freq_deviation_timing"):
    style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 4.8), dpi=100)

    fp.event_lines(ax1, ytxt=-1.22)
    ax1.axhline(-1.0, color="black", lw=1.2)
    ax1.text(7.5, -0.98, "UFLS stage-1 ($\\Delta f$ = −1.0 Hz)",
             ha="center", va="bottom", fontsize=9)
    for key, case, tc, c, ls, lw, lab in fp.CASES:
        r = fp.run(key, case, tc)
        mt = fm.metrics(r)
        ax1.plot(r["t"], r["f"] - fm.F0, color=c, ls=ls, lw=lw,
                 label=f"{lab} ($\\Delta f_{{nadir}}$ = {mt['nadir'] - fm.F0:.2f} Hz)")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-1.4, 0.1)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Frequency Deviation $\\Delta f$ (Hz)", fontsize=12)
    ax1.set_title("(a) IEEE 39-Bus: Control Comparison", fontsize=11.5)
    ax1.grid(True, color="0.8", lw=0.8)
    ax1.legend(loc="lower right", fontsize=8.4, edgecolor="black",
               fancybox=False, framealpha=1)

    fp.event_lines(ax2, ytxt=-1.22)
    cols = {"IEEE 9-Bus": "#1f6fc5", "IEEE 39-Bus": "#2e8b3a",
            "IEEE 118-Bus": "#d62728"}
    pfs = []
    for s, H in fm.SYSTEMS.items():
        nc = fp.run("NC", "NC", 0.0, s)
        pf = fp.run("PF", "PF", 0.0, s)
        pfs.append(pf["f"] - fm.F0)
        mt = fm.metrics(pf)
        ax2.plot(nc["t"], nc["f"] - fm.F0, color=cols[s], lw=1.0, ls="--",
                 alpha=0.6)
        ax2.plot(pf["t"], pf["f"] - fm.F0, color=cols[s], lw=2.0,
                 label=f"{s}, $H_{{sys}}$={H} s (CNN-LSTM + BESS, "
                       f"$\\Delta f_{{nadir}}$ = {mt['nadir'] - fm.F0:.2f} Hz)")
    pfs = np.array(pfs)
    ax2.fill_between(pf["t"], pfs.min(0), pfs.max(0), color="0.6",
                     alpha=0.18, lw=0, label="Inter-system variation band")
    ax2.plot([], [], color="0.4", lw=1.0, ls="--",
             label="Same systems, no control (reference)")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-1.4, 0.1)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Frequency Deviation $\\Delta f$ (Hz)", fontsize=12)
    ax2.set_title("(b) CNN-LSTM + BESS Across Test Systems "
                  "(10.3% generation loss)", fontsize=11.5)
    ax2.grid(True, color="0.8", lw=0.8)
    ax2.legend(loc="lower right", fontsize=8.0, edgecolor="black",
               fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
