"""Two-panel rotor-angle figure: (a) stable G1/G4/G7, (b) G10 under no
control (NC), conventional control (CC) and CNN-LSTM + STATCOM (PF),
re-timed to the measured latency budget.

Fault onset 0.1 s, cleared 0.25 s. Decision available 519.2 ms after onset
(0.619 s); STATCOM action at 0.649 s + T_comm. Before that instant the PF
trajectory is identical to NC (a rotor angle cannot jump, so the support
acts through the swing equation).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

T_FAULT, T_CLEAR = 0.10, 0.25
T_WIN, T_PRE, T_INF = 0.500, 0.0019, 0.0173
T_DEC = T_FAULT + T_WIN + T_PRE + T_INF            # 0.6192 s
TAU_STATCOM = 0.030
T_STATCOM = T_DEC + TAU_STATCOM                    # 0.6492 s

DT = 1e-3
t = np.arange(0.0, 4.0 + DT, DT)


# ------------------------------------------------------------ panel (a)
def stable(base, amp, sigma, period):
    y = np.full_like(t, base)
    m = t >= T_CLEAR
    tau = t[m] - T_CLEAR
    y[m] += amp * np.exp(-sigma * tau) * np.sin(2 * np.pi * tau / period)
    return y


g1 = stable(20.0, 35.0, 0.82, 1.45)
g4 = stable(25.0, 42.6, 0.66, 1.55)
g7 = stable(28.0, 46.0, 0.49, 1.60)

# ------------------------------------------------------------ panel (b)
D0 = np.deg2rad(19.0)
PM = 5.0
P_POST = {"NC": 1.0, "CC": 3.0}      # post-disturbance capability
P_CTRL = PM / np.sin(np.deg2rad(22.0))  # with STATCOM support
D_U, D_C = 0.25, 3.0


def g10(case, t_act=None):
    p_post = P_POST["NC" if case == "PF" else case]

    def rhs(tt, x):
        if tt < T_CLEAR:
            return [0.0, 0.0]
        k = 0.0
        if t_act is not None and tt >= t_act:
            k = 1.0 - np.exp(-(tt - t_act) / (TAU_STATCOM / 3.0))
        pmax = p_post + (P_CTRL - p_post) * k
        d = D_U + (D_C - D_U) * k
        return [x[1], PM - pmax * np.sin(x[0]) - d * x[1]]

    sol = solve_ivp(rhs, (0, t[-1]), [D0, 0.0], t_eval=t, max_step=1e-3,
                    rtol=1e-8, atol=1e-10)
    return np.rad2deg(sol.y[0])


def plot(out="g10_nc_cc_pf_timing"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    })
    tc_late = 0.200
    nc, cc = g10("NC"), g10("CC")
    pf0 = g10("PF", T_STATCOM)
    pf2 = g10("PF", T_STATCOM + tc_late)
    t_nc = t[np.argmax(nc > 180)]
    t_cc = t[np.argmax(cc > 180)]
    i0, i2 = np.argmax(pf0), np.argmax(pf2)
    print(f"NC loses sync at {t_nc:.2f} s, CC at {t_cc:.2f} s")
    print(f"PF T_comm=0 peak {pf0[i0]:.1f} deg at {t[i0]:.2f} s; "
          f"T_comm=200 ms peak {pf2[i2]:.1f} deg at {t[i2]:.2f} s")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 8.0), dpi=100)
    for ax in (ax1, ax2):
        ax.axvline(T_FAULT, color="black", ls=":", lw=1.1)
        ax.axvline(T_CLEAR, color="black", ls="--", lw=1.1)
        ax.grid(True, color="0.85", lw=0.8)
        ax.set_xlim(-0.2, 4.2)

    ax1.plot(t, g1, color="#1f6fb4", lw=2.2, label="G1 ($H$=5.5 s, stable)")
    ax1.plot(t, g4, color="#228b3b", lw=2.2, ls="--",
             label="G4 ($H$=4.8 s, stable)")
    ax1.plot(t, g7, color="#8b6508", lw=2.2, ls="-.",
             label="G7 ($H$=4.5 s, stable)")
    ax1.set_ylim(0, 80)
    ax1.set_ylabel("Rotor angle $\\delta$ (degrees)", fontsize=12)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_title("(a) Stable Generators G1, G4 and G7", fontsize=11.5)
    ax1.legend(loc="upper right", fontsize=9, edgecolor="black",
               fancybox=False, framealpha=1)

    green = "#1e9e55"
    ax2.axvspan(T_FAULT, T_FAULT + T_WIN, color="0.5", alpha=0.08, lw=0)
    ax2.axvline(T_DEC, color="0.3", lw=0.9)
    ax2.axvline(T_STATCOM, color="0.3", lw=0.9)
    ax2.plot(t, nc, color="#c0392b", ls="--", lw=2.2,
             label=f"G10 — NC (loses sync at $t$≈{t_nc:.2f} s)")
    ax2.plot(t, cc, color="#e67e22", ls="-.", lw=2.2,
             label=f"G10 — CC (loses sync at $t$≈{t_cc:.2f} s)")
    ax2.plot(t, pf0, color=green, lw=2.8,
             label=f"G10 — PF (CNN-LSTM+STATCOM, $T_{{comm}}$=0, "
                   f"$\\delta_{{peak}}$={pf0[i0]:.1f}°)")
    ax2.plot(t, pf2, color="0.35", lw=2.0, ls=(0, (1, 1.2)),
             label=f"G10 — PF ($T_{{comm}}$={tc_late*1e3:.0f} ms, "
                   f"$\\delta_{{peak}}$={pf2[i2]:.1f}°)")
    ax2.axhline(180, color="black", ls=":", lw=1.3,
                label="180° stability boundary")
    ax2.axhline(pf0[i0], color=green, ls=":", lw=0.8)
    ax2.annotate(f"$\\delta_{{peak}}$={pf0[i0]:.1f}°",
                 xy=(t[i0], pf0[i0]), xytext=(t[i0] + 0.55, pf0[i0] + 30),
                 color=green, fontsize=10,
                 arrowprops=dict(arrowstyle="->", color=green, lw=0.9))
    ax2.text(T_DEC - 0.03, 125, f"Decision ($t$={T_DEC:.3f} s)", rotation=90,
             ha="right", va="center", fontsize=8.5)
    ax2.text(T_STATCOM + 0.03, 125, f"STATCOM action ($t$={T_STATCOM:.3f} s)",
             rotation=90, ha="left", va="center", fontsize=8.5)
    ax2.text(T_FAULT + T_WIN / 2, 5, "PMU window", ha="center", va="bottom",
             fontsize=8.5, bbox=dict(fc="white", ec="none", pad=0.4))
    ax2.set_ylim(0, 205)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Rotor angle $\\delta$ (degrees)", fontsize=12)
    ax2.set_title("(b) G10 — With and Without CNN-LSTM Corrective Control",
                  fontsize=11.5)
    ax2.legend(loc="upper right", fontsize=8.4, edgecolor="black",
               fancybox=False, framealpha=1)

    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
