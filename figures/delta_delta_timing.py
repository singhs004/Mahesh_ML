"""Rotor-angle deviation (delta - delta0) of G1/G4/G7 and G10 from the
shared model in g10_model.py (same curves as rotor_angle_timing.py)."""
import numpy as np
import matplotlib.pyplot as plt

import g10_model as m
import style


def plot(out="delta_delta_timing"):
    style.apply()
    rng = np.random.default_rng(11)
    nz = lambda s=0.3: rng.normal(0, s, m.t.size)
    gens = m.stable_gens()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 8.0), dpi=100)

    style.event_lines(ax1, m, decision=False)
    for (name, y), c, mk, off in zip(gens.items(),
                                     ["#1f6fc5", "#2e8b3a", "#a0179e"],
                                     ["o", "s", "^"], [50, 20, 0]):
        ax1.plot(m.t, y - y[0] + nz(0.08), color=c, lw=1.5, marker=mk, ms=6,
                 markevery=(off, 200), label=name)
    ax1.set_xlim(0, m.T_END)
    ax1.set_ylim(-15, 20)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Δδ (degrees)")
    ax1.set_title("(a) Stable generators", fontsize=11)
    ax1.grid(True, color="0.8", lw=0.8)
    ax1.legend(loc="upper right", fontsize=9.5, edgecolor="black",
               fancybox=False, framealpha=1)

    nc, _ = m.g10("T")
    st0, _ = m.g10("T", "STATCOM", 0.0)
    st2, _ = m.g10("T", "STATCOM", 0.2)
    sv0, _ = m.g10("T", "SVC", 0.0)
    d0 = m.DELTA0
    style.event_lines(ax2, m, labels=True, y=120, devices=("STATCOM", "SVC"))
    for y, c, mk, ls, off, lab in [
            (nc, "#d62728", "o", "-", 0, "G10 — No control (unstable)"),
            (st0, "black", "s", "-", 120,
             f"G10 — CNN-LSTM+STATCOM ($T_{{comm}}$=0, Δδ$_{{peak}}$={st0.max()-d0:.1f}°)"),
            (st2, "0.4", "D", "--", 60,
             f"G10 — CNN-LSTM+STATCOM ($T_{{comm}}$=200 ms, Δδ$_{{peak}}$={st2.max()-d0:.1f}°)"),
            (sv0, "#e07b00", "X", "-.", 180,
             f"G10 — CNN-LSTM+SVC ($T_{{comm}}$=0, Δδ$_{{peak}}$={sv0.max()-d0:.1f}°)")]:
        ax2.plot(m.t, y - d0 + nz(), color=c, ls=ls, lw=1.5, marker=mk, ms=6,
                 markevery=(off, 250), label=lab, zorder=3)
    ax2.axhline(180 - d0, color="black", ls="--", lw=1.0)
    ax2.text(m.T_END - 0.05, 180 - d0 + 3, "Loss of synchronism (δ = 180°)",
             ha="right", va="bottom", fontsize=8.5)
    ax2.set_xlim(0, m.T_END)
    ax2.set_ylim(-10, 190)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Δδ (degrees)")
    ax2.set_title("(b) G10", fontsize=11)
    ax2.grid(True, color="0.8", lw=0.8)
    ax2.legend(loc="center right", fontsize=7.6, edgecolor="black",
               fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
