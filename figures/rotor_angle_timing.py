"""Rotor angle of G1/G4/G7 and G10 (absolute), CNN-LSTM + STATCOM / SVC,
from the shared model in g10_model.py."""
import numpy as np
import matplotlib.pyplot as plt

import g10_model as m
import style


def plot(out="rotor_angle_timing"):
    style.apply()
    rng = np.random.default_rng(7)
    nz = lambda: rng.normal(0, 0.15, m.t.size)
    gens = m.stable_gens()
    nc, _ = m.g10("T")
    st0, _ = m.g10("T", "STATCOM", 0.0)
    st2, _ = m.g10("T", "STATCOM", 0.2)
    sv0, _ = m.g10("T", "SVC", 0.0)

    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=100)
    style.event_lines(ax, m, labels=True, y=150, devices=("STATCOM", "SVC"))
    series = [
        (gens["G1"], "#1f6fc5", "o", "-", 0, "G1 ($H$=5.5 s, stable)"),
        (gens["G4"], "#2e8b3a", "s", "-", 70, "G4 ($H$=4.8 s, stable)"),
        (gens["G7"], "#a0179e", "^", "-", 140, "G7 ($H$=4.5 s, stable)"),
        (nc, "#d62728", "v", "-", 0, "G10 ($H$=3.5 s), no control"),
        (st0, "black", "D", "-", 105,
         f"G10, CNN-LSTM+STATCOM ($T_{{comm}}$=0, $\\delta_{{peak}}$={st0.max():.1f}°)"),
        (st2, "0.4", "P", "--", 105,
         f"G10, CNN-LSTM+STATCOM ($T_{{comm}}$=200 ms, $\\delta_{{peak}}$={st2.max():.1f}°)"),
        (sv0, "#e07b00", "X", "-.", 105,
         f"G10, CNN-LSTM+SVC ($T_{{comm}}$=0, $\\delta_{{peak}}$={sv0.max():.1f}°)"),
    ]
    for y, c, mk, ls, off, lab in series:
        ax.plot(m.t, y + nz(), color=c, ls=ls, lw=1.6, marker=mk, ms=6,
                markevery=(off, 250), label=lab, zorder=3)
    ax.axhline(180, color="black", ls="--", lw=1.0)
    ax.set_xlim(0, m.T_END)
    ax.set_ylim(0, 200)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Rotor angle $\\delta$ (degrees)", fontsize=12)
    ax.grid(True, color="0.8", lw=0.8)
    ax.legend(loc="upper right", fontsize=8.2, edgecolor="black",
              fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
