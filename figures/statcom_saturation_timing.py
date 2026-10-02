"""(a) G10 rotor angle and (b) Bus 16 STATCOM output with saturation, from
the shared model in g10_model.py. Case T is the same G10 contingency as in
every other rotor-angle figure; the combined disturbance is more severe and
asks for more reactive power than the 150 Mvar rating."""
import numpy as np
import matplotlib.pyplot as plt

import g10_model as m
import style


def plot(out="statcom_saturation_timing"):
    style.apply()
    ncc, _ = m.g10("combined")
    yt, qt = m.g10("T", "STATCOM")
    yc, qc = m.g10("combined", "STATCOM")
    yc_unc, qc_unc = m.g10("combined", "STATCOM", cap=1e9)
    it, ic = np.argmax(yt), np.argmax(yc)
    sat = np.where(qc >= 0.98 * m.Q_RATED)[0]
    s_on, s_off = m.t[sat[0]], m.t[sat[-1]]
    green, purple = "#1e9e55", "#8e44ad"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 7.6), dpi=100)
    for ax in (ax1, ax2):
        style.event_lines(ax, m, devices=("STATCOM",))
        ax.grid(True, color="0.85", lw=0.8)
        ax.set_xlim(0, m.T_END)

    ax1.plot(m.t, ncc, color="#c0392b", ls="--", lw=2.2,
             label="NC (combined disturbance) — G10 loses synchronism")
    ax1.plot(m.t, yt, color=green, ls="-.", lw=2.4,
             label=f"PF unsaturated (Case T alone, $\\delta_{{peak}}$={yt.max():.1f}°)")
    ax1.plot(m.t, yc, color=purple, lw=2.8,
             label=f"PF saturated (combined disturbance, $\\delta_{{peak}}$={yc.max():.1f}°)")
    ax1.plot(m.t, yc_unc, color=purple, lw=1.2, ls=":",
             label=f"Combined, if unlimited rating ($\\delta_{{peak}}$={yc_unc.max():.1f}°)")
    ax1.axhline(180, color="black", ls="--", lw=1.3,
                label="180° stability boundary")
    ax1.annotate(f"{180 - yc.max():.0f}° margin", xy=(m.t[ic], yc.max()),
                 xytext=(m.t[ic] + 0.5, yc.max() + 30), color=purple,
                 fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color=purple, lw=0.9))
    ax1.text(m.T_DEC - 0.02, 125, f"Decision ({m.T_DEC:.3f} s)", rotation=90,
             ha="right", va="center", fontsize=8.3)
    ax1.set_ylim(0, 200)
    ax1.set_ylabel("G10 rotor angle $\\delta$ (degrees)", fontsize=12)
    ax1.set_title("(a) G10 Rotor Angle — Effect of STATCOM Saturation",
                  fontsize=11.5)
    ax1.legend(loc="upper right", fontsize=8.2, edgecolor="black",
               fancybox=False, framealpha=1)

    ax2.axvspan(s_on, s_off, color="#f5cba7", alpha=0.45, lw=0,
                label=f"Saturation window ({s_on:.3f}–{s_off:.2f} s)")
    ax2.plot(m.t, qt, color=green, ls="-.", lw=2.4,
             label=f"Case T alone (peak {qt.max():.0f} Mvar, no saturation)")
    ax2.plot(m.t, qc, color=purple, lw=2.8,
             label="Combined disturbance (capped at 150 Mvar)")
    ax2.plot(m.t, np.where(qc_unc > m.Q_RATED, qc_unc, np.nan), color=purple,
             ls=":", lw=1.4, label="Combined-disturbance demand (unclipped)")
    ax2.axhline(m.Q_RATED, color="black", ls="--", lw=1.3,
                label="150 Mvar rated capacity limit")
    ax2.axhline(0.9 * m.Q_RATED, color="0.4", ls=":", lw=1.0,
                label="90% utilisation threshold")
    ax2.set_ylim(0, 230)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("$Q_{FACTS}$ — STATCOM Bus 16 (Mvar)", fontsize=12)
    ax2.set_title("(b) STATCOM Reactive-Power Output — Saturation "
                  "Clamping Visible", fontsize=11.5)
    ax2.legend(loc="upper right", fontsize=8.2, edgecolor="black",
               fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
