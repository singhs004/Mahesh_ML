"""(a) G1/G4/G7 and (b) G10 under NC, CC and CNN-LSTM + STATCOM (PF), from
the shared model in g10_model.py."""
import numpy as np
import matplotlib.pyplot as plt

import g10_model as m
import style


def plot(out="g10_nc_cc_pf_timing"):
    style.apply()
    gens = m.stable_gens()
    nc, _ = m.g10("T")
    cc, _ = m.g10("CC")
    pf0, _ = m.g10("T", "STATCOM", 0.0)
    pf2, _ = m.g10("T", "STATCOM", 0.2)
    t_nc, t_cc = m.cross_time(nc), m.cross_time(cc)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 8.0), dpi=100)
    style.event_lines(ax1, m, decision=False)
    for (name, y), c, ls, h in zip(gens.items(),
                                   ["#1f6fb4", "#228b3b", "#8b6508"],
                                   ["-", "--", "-."], [5.5, 4.8, 4.5]):
        ax1.plot(m.t, y, color=c, lw=2.2, ls=ls,
                 label=f"{name} ($H$={h} s, stable)")
    ax1.set_xlim(0, m.T_END)
    ax1.set_ylim(0, 80)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Rotor angle $\\delta$ (degrees)", fontsize=12)
    ax1.set_title("(a) Stable Generators G1, G4 and G7", fontsize=11.5)
    ax1.grid(True, color="0.85", lw=0.8)
    ax1.legend(loc="upper right", fontsize=9, edgecolor="black",
               fancybox=False, framealpha=1)

    green = "#1e9e55"
    style.event_lines(ax2, m, labels=True, y=125, devices=("STATCOM",))
    ax2.plot(m.t, nc, color="#c0392b", ls="--", lw=2.2,
             label=f"G10 — NC (loses sync at $t$≈{t_nc:.2f} s)")
    ax2.plot(m.t, cc, color="#e67e22", ls="-.", lw=2.2,
             label=f"G10 — CC (loses sync at $t$≈{t_cc:.2f} s)")
    ax2.plot(m.t, pf0, color=green, lw=2.8,
             label=f"G10 — PF (CNN-LSTM+STATCOM, $T_{{comm}}$=0, "
                   f"$\\delta_{{peak}}$={pf0.max():.1f}°)")
    ax2.plot(m.t, pf2, color="0.35", lw=2.0, ls=(0, (1, 1.2)),
             label=f"G10 — PF ($T_{{comm}}$=200 ms, "
                   f"$\\delta_{{peak}}$={pf2.max():.1f}°)")
    ax2.axhline(180, color="black", ls=":", lw=1.3,
                label="180° stability boundary")
    ax2.set_xlim(0, m.T_END)
    ax2.set_ylim(0, 205)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Rotor angle $\\delta$ (degrees)", fontsize=12)
    ax2.set_title("(b) G10 — With and Without CNN-LSTM Corrective Control",
                  fontsize=11.5)
    ax2.grid(True, color="0.85", lw=0.8)
    ax2.legend(loc="upper right", fontsize=8.4, edgecolor="black",
               fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
