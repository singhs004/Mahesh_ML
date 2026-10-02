"""Rotor-angle response of G10 with the CNN-LSTM + STATCOM/SVC control chain
re-timed to the measured latency budget.

Latency budget (fault onset at t = 0.1 s):
    PMU observation window        500 ms
    Preprocessing                 1.9 ms
    CNN-LSTM inference (CPU)      17.3 ms
    Decision available            519.2 ms after onset
    STATCOM response              < 30 ms   -> action at ~549 ms + T_comm
    SVC response                  ~150 ms   -> action at ~669 ms + T_comm
Before the corrective action the controlled G10 trajectory is identical to the
uncontrolled one; afterwards a swing-equation model with the compensator
output ramping in over its response time is integrated.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------- timing
T_FAULT = 0.10
T_CLEAR = 0.25
T_WIN, T_PRE, T_INF = 0.500, 0.0019, 0.0173
T_DEC = T_FAULT + T_WIN + T_PRE + T_INF            # 0.6192 s
TAU_STATCOM, TAU_SVC = 0.030, 0.150
T_STATCOM = T_DEC + TAU_STATCOM                    # ~0.649 s (+T_comm)
T_SVC = T_DEC + TAU_SVC                            # ~0.769 s (+T_comm)

DT = 1e-3
t = np.arange(0.0, 3.0 + DT, DT)
rng = np.random.default_rng(7)
D0 = 11.5                                          # pre-fault angle (deg)


def noise(scale=0.45):
    return rng.normal(0.0, scale, t.size)


# ------------------------------------------------- stable generators G1/G4/G7
def stable_gen(amp, f, zeta, phase_lag):
    y = np.full_like(t, D0)
    m = t >= T_FAULT
    tt = t[m] - T_FAULT
    wn = 2 * np.pi * f
    y[m] += amp * np.exp(-zeta * wn * tt) * np.sin(wn * tt + phase_lag) \
        - amp * np.sin(phase_lag) * np.exp(-tt / 0.05)
    return y


g1 = stable_gen(11.0, 1.15, 0.30, 0.0)
g4 = stable_gen(14.0, 1.00, 0.28, 0.25)
g7 = stable_gen(10.0, 1.05, 0.25, 0.10)


# ------------------------------------------------ G10 uncontrolled (unstable)
def g10_uncontrolled(tt):
    sig = 1.0 / (1.0 + np.exp(-(tt - 0.86) / 0.135))
    sig0 = 1.0 / (1.0 + np.exp(-(0.0 - 0.86) / 0.135))
    bump = 5.0 * np.exp(-((tt - 0.2) / 0.07) ** 2)
    return D0 + 175.0 * (sig - sig0) / (1 - sig0) + bump


def g10_uncontrolled_rate(tt, h=1e-4):
    return (g10_uncontrolled(tt + h) - g10_uncontrolled(tt - h)) / (2 * h)


g10_nc = g10_uncontrolled(t)

# --------------------------------------- G10 with corrective action (swing eq)
# Normalised SMIB: delta'' = Pm - Pmax(t) sin(delta) - D(t) delta'   [rad, s]
WN, ZETA = 6.0, 0.50
DEQ = np.deg2rad(12.0)
PMAX_C = WN ** 2 / np.cos(DEQ)          # compensated post-fault capability
PM = PMAX_C * np.sin(DEQ)
PMAX_U = 0.80 * PM                      # uncompensated: no equilibrium
D_C, D_U = 2 * ZETA * WN, 0.3


def g10_controlled(t_act, tau):
    """Trajectory of G10 when the compensator acts at t_act with lag tau."""
    y = g10_nc.copy()
    m = t >= t_act
    x0 = [np.deg2rad(g10_uncontrolled(t_act)),
          np.deg2rad(g10_uncontrolled_rate(t_act))]

    def rhs(tt, x):
        k = 1.0 - np.exp(-(tt - t_act) / (tau / 3.0))   # ~95 % in tau
        pmax = PMAX_U + (PMAX_C - PMAX_U) * k
        d = D_U + (D_C - D_U) * k
        return [x[1], PM - pmax * np.sin(x[0]) - d * x[1]]

    sol = solve_ivp(rhs, (t_act, t[-1]), x0, t_eval=t[m], max_step=1e-3,
                    rtol=1e-8, atol=1e-10)
    y[m] = np.rad2deg(sol.y[0])
    return y


def is_stable(traj):
    return traj.max() < 180.0


def plot(out="rotor_angle_timing"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    })
    tc_late = 0.200
    g10_st0 = g10_controlled(T_STATCOM, TAU_STATCOM)
    g10_sv0 = g10_controlled(T_SVC, TAU_SVC)
    g10_st2 = g10_controlled(T_STATCOM + tc_late, TAU_STATCOM)

    fig, ax = plt.subplots(figsize=(7.5, 4.9), dpi=100)
    ax.axvspan(T_FAULT, T_FAULT + T_WIN, color="0.5", alpha=0.08, lw=0)

    me = 210
    series = [
        (g1, "#1f6fc5", "o", "-", 0, "G1 (H=5.5, Stable)"),
        (g4, "#2e8b3a", "s", "-", 70, "G4 (H=4.8, Stable)"),
        (g7, "#a0179e", "^", "-", 140, "G7 (H=4.5, Stable)"),
        (g10_nc, "#d62728", "v", "-", 0, "G10 (H=3.5, Unstable \u2014 No Control)"),
        (g10_st0, "black", "D", "-", 105,
         "G10, CNN-LSTM+STATCOM ($T_{comm}$=0, act. 0.649 s)"),
        (g10_st2, "0.35", "P", "--", 105,
         f"G10, CNN-LSTM+STATCOM ($T_{{comm}}$={tc_late*1e3:.0f} ms, act. "
         f"{T_STATCOM + tc_late:.3f} s)"),
        (g10_sv0, "#e07b00", "X", "-.", 105,
         "G10, CNN-LSTM+SVC ($T_{comm}$=0, act. 0.769 s)"),
    ]
    for y, c, mk, ls, off, lab in series:
        ax.plot(t, y + noise(), color=c, ls=ls, lw=1.6, marker=mk, ms=7,
                markevery=(off, me), label=lab, zorder=3)

    def vline(x, text, side="left"):
        ax.axvline(x, color="black", lw=1.0, zorder=2)
        dx = -0.012 if side == "left" else 0.012
        ax.text(x + dx, 197, text, rotation=90, va="top",
                ha="right" if side == "left" else "left", fontsize=9.5)

    vline(T_FAULT, f"Fault Onset (t={T_FAULT:g} s)")
    vline(T_CLEAR, f"Fault Cleared (t={T_CLEAR:g} s)")
    vline(T_DEC, f"Decision Available (t={T_DEC:.3f} s)")
    vline(T_STATCOM, f"STATCOM Action (t={T_STATCOM:.3f} s)", side="right")
    vline(T_SVC, f"SVC Action (t={T_SVC:.3f} s)", side="right")
    ax.annotate("", xy=(T_FAULT, -5), xytext=(T_FAULT + T_WIN, -5),
                arrowprops=dict(arrowstyle="<->", lw=0.9))
    ax.text(T_FAULT + T_WIN / 2, -3.8, "PMU window (500 ms)", ha="center",
            va="bottom", fontsize=8.5,
            bbox=dict(fc="white", ec="none", pad=0.5))

    ax.set_xlim(0, 3.0)
    ax.set_ylim(-10, 200)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Rotor Angle \u03b4 (degrees)", fontsize=12)
    ax.grid(True, color="0.75", lw=0.8)
    ax.legend(loc="center right", bbox_to_anchor=(0.995, 0.47), fontsize=8.6,
              frameon=True, edgecolor="black", fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")
    return fig


if __name__ == "__main__":
    for tc in np.arange(0.0, 0.51, 0.05):
        s = g10_controlled(T_STATCOM + tc, TAU_STATCOM)
        v = g10_controlled(T_SVC + tc, TAU_SVC)
        print(f"T_comm={tc*1e3:4.0f} ms  STATCOM peak={s.max():7.1f} "
              f"({'stable' if is_stable(s) else 'UNSTABLE'})  SVC peak="
              f"{v.max():7.1f} ({'stable' if is_stable(v) else 'UNSTABLE'})")
    plot()
