"""Two-panel rotor-angle deviation figure (stable G1/G4/G7 and G10) with the
CNN-LSTM corrective action re-timed to the measured latency budget.

Fault onset 0.1 s, cleared 0.25 s. Decision available 519.2 ms after onset
(0.619 s); STATCOM action at 0.649 s + T_comm, SVC action at 0.769 s + T_comm.
Until the action instant the controlled G10 trajectory equals the
uncontrolled one; afterwards a swing-equation model is integrated with the
compensator output ramping in over its response time.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

T_FAULT, T_CLEAR = 0.10, 0.25
T_WIN, T_PRE, T_INF = 0.500, 0.0019, 0.0173
T_DEC = T_FAULT + T_WIN + T_PRE + T_INF            # 0.6192 s
TAU_STATCOM, TAU_SVC = 0.030, 0.150
T_STATCOM = T_DEC + TAU_STATCOM                    # 0.6492 s
T_SVC = T_DEC + TAU_SVC                            # 0.7692 s

DT = 1e-3
rng = np.random.default_rng(11)


def noise(t, scale=0.35):
    return rng.normal(0.0, scale, t.size)


# ------------------------------------------------------------ top panel
t1 = np.arange(0.0, 2.5 + DT, DT)


def stable_dev(amp, f, zeta, phi, rise):
    y = np.zeros_like(t1)
    m = t1 >= T_FAULT
    tt = t1[m] - T_FAULT
    wn = 2 * np.pi * f
    y[m] = amp * np.exp(-zeta * wn * tt) * \
        (np.sin(wn * tt + phi) - np.sin(phi) * np.exp(-tt / rise))
    return y


g1 = stable_dev(14.5, 1.45, 0.11, 0.95, 0.025)
g4 = stable_dev(13.5, 1.35, 0.10, 1.05, 0.018)
g7 = stable_dev(11.0, 2.10, 0.12, 0.70, 0.010)

# ------------------------------------------------------------ bottom panel
t2 = np.arange(0.0, 4.0 + DT, DT)
D0 = 12.0                                           # pre-fault angle (deg)


def g10_nc_dev(tt):
    tt = np.asarray(tt, dtype=float)
    early = np.where(tt > T_FAULT,
                     9.0 * (1 - np.exp(-(tt - T_FAULT) / 0.08)), 0.0)
    s = 1 / (1 + np.exp(-(tt - 1.12) / 0.17))
    s0 = 1 / (1 + np.exp(1.12 / 0.17))
    return early + 164.0 * (s - s0) / (1 - s0)


def g10_nc_rate(tt, h=1e-4):
    return (g10_nc_dev(tt + h) - g10_nc_dev(tt - h)) / (2 * h)


WN, ZETA = 6.0, 0.50
DEQ = np.deg2rad(D0)
PMAX_C = WN ** 2 / np.cos(DEQ)
PM = PMAX_C * np.sin(DEQ)
PMAX_U = 0.80 * PM
D_C, D_U = 2 * ZETA * WN, 0.3


def g10_ctrl_dev(t_act, tau):
    y = g10_nc_dev(t2)
    m = t2 >= t_act
    x0 = [np.deg2rad(D0 + g10_nc_dev(t_act)), np.deg2rad(g10_nc_rate(t_act))]

    def rhs(tt, x):
        k = 1.0 - np.exp(-(tt - t_act) / (tau / 3.0))
        pmax = PMAX_U + (PMAX_C - PMAX_U) * k
        d = D_U + (D_C - D_U) * k
        return [x[1], PM - pmax * np.sin(x[0]) - d * x[1]]

    sol = solve_ivp(rhs, (t_act, t2[-1]), x0, t_eval=t2[m], max_step=1e-3,
                    rtol=1e-8, atol=1e-10)
    y[m] = np.rad2deg(sol.y[0]) - D0
    return y


def plot(out="delta_delta_timing"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 7.6), dpi=100)

    for y, c, mk, off, lab in [(g1, "#1f6fc5", "o", 50, "G1"),
                               (g4, "#2e8b3a", "s", 20, "G4"),
                               (g7, "#a0179e", "^", 0, "G7")]:
        ax1.plot(t1, y + noise(t1, 0.25), color=c, lw=1.5, marker=mk, ms=6,
                 markevery=(off, 160), label=lab)
    for x in (T_FAULT, T_CLEAR):
        ax1.axvline(x, color="black", lw=1.0)
    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(-16, 22)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Δδ (degrees)")
    ax1.grid(True, color="0.75", lw=0.8)
    ax1.legend(loc="upper right", fontsize=9.5, edgecolor="black",
               fancybox=False, framealpha=1)

    tc_late = 0.200
    st0 = g10_ctrl_dev(T_STATCOM, TAU_STATCOM)
    st2 = g10_ctrl_dev(T_STATCOM + tc_late, TAU_STATCOM)
    sv0 = g10_ctrl_dev(T_SVC, TAU_SVC)
    series = [
        (g10_nc_dev(t2), "#d62728", "o", "-", 0,
         "G10 — No Control (Unstable)"),
        (st0, "black", "s", "-", 120,
         f"G10 — CNN-LSTM+STATCOM ($T_{{comm}}$=0, act. {T_STATCOM:.3f} s)"),
        (st2, "0.4", "D", "--", 60,
         f"G10 — CNN-LSTM+STATCOM ($T_{{comm}}$={tc_late*1e3:.0f} ms, "
         f"act. {T_STATCOM + tc_late:.3f} s)"),
        (sv0, "#e07b00", "X", "-.", 180,
         f"G10 — CNN-LSTM+SVC ($T_{{comm}}$=0, act. {T_SVC:.3f} s)"),
    ]
    for y, c, mk, ls, off, lab in series:
        ax2.plot(t2, y + noise(t2), color=c, ls=ls, lw=1.5, marker=mk, ms=6,
                 markevery=(off, 250), label=lab, zorder=3)
        print(f"{lab:70s} peak {y.max():6.1f} deg")

    def vline(x, text, side):
        ax2.axvline(x, color="black", lw=1.0, zorder=2)
        dx = -0.025 if side == "left" else 0.025
        ax2.text(x + dx, 186, text, rotation=90, va="top", fontsize=8.5,
                 ha="right" if side == "left" else "left")

    vline(T_FAULT, "Fault Onset (0.1 s)", "right")
    vline(T_CLEAR, "Fault Cleared (0.25 s)", "right")
    vline(T_DEC, f"Decision ({T_DEC:.3f} s)", "left")
    vline(T_STATCOM, f"STATCOM Action ({T_STATCOM:.3f} s)", "right")
    vline(T_SVC, f"SVC Action ({T_SVC:.3f} s)", "right")

    ax2.set_xlim(0, 4.0)
    ax2.set_ylim(-10, 190)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Δδ (degrees)")
    ax2.grid(True, color="0.75", lw=0.8)
    ax2.legend(loc="center right", bbox_to_anchor=(0.995, 0.42), fontsize=7.6,
               edgecolor="black", fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    for tc in np.arange(0.0, 0.51, 0.05):
        s = g10_ctrl_dev(T_STATCOM + tc, TAU_STATCOM).max()
        v = g10_ctrl_dev(T_SVC + tc, TAU_SVC).max()
        f = lambda p: "stable" if p < 168 else "UNSTABLE"
        print(f"T_comm={tc*1e3:4.0f} ms STATCOM {s:6.1f} ({f(s)})  "
              f"SVC {v:6.1f} ({f(v)})")
    plot()
