"""Common matplotlib style for all figures."""
import matplotlib.pyplot as plt


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    })


def event_lines(ax, m, decision=True, labels=False, y=None, devices=()):
    ax.axvline(m.T_FAULT, color="black", ls=":", lw=1.1)
    ax.axvline(m.T_CLEAR, color="black", ls="--", lw=1.1)
    ax.axvspan(m.T_FAULT, m.T_FAULT + m.T_WIN, color="0.5", alpha=0.08, lw=0)
    if decision:
        ax.axvline(m.T_DEC, color="0.3", lw=0.9)
    for d in devices:
        ax.axvline(m.T_ACT[d], color="0.3", lw=0.9, ls="-.")
    if labels and y is not None:
        items = [(m.T_FAULT, f"Fault onset ({m.T_FAULT:g} s)", "l"),
                 (m.T_CLEAR, f"Fault cleared ({m.T_CLEAR:g} s)", "r")]
        if decision:
            items.append((m.T_DEC, f"Decision ({m.T_DEC:.3f} s)", "l"))
        for d in devices:
            items.append((m.T_ACT[d], f"{d} action ({m.T_ACT[d]:.3f} s)", "r"))
        for x, txt, side in items:
            ax.text(x + (-0.02 if side == "l" else 0.02), y, txt, rotation=90,
                    va="center", ha="right" if side == "l" else "left",
                    fontsize=8.3, bbox=dict(fc="white", ec="none", pad=0.2))
