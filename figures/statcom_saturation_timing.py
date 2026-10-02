"""G10 rotor angle and Bus 16 STATCOM output under saturation, re-timed to
the measured latency budget.

Disturbance at t = 0.25 s. Decision available 519.2 ms later (t = 0.769 s);
the STATCOM reaches its command within 30 ms (t = 0.799 s). Before the
decision the STATCOM output is zero and G10 follows its uncontrolled path.

Normalised swing model (rad, s):
    delta'' = PM - Pmax(t) sin(delta) - D(t) delta'
    Pmax(t) = Pnet(t) + EFF * (Q / Q_REQ) * (P_TARGET - P_DIST)
Pnet drops at the disturbance and the network recovers slowly from
T_REC onwards. The STATCOM command follows the remaining gap, scaled to
the case's peak requirement Q_REQ, and is clipped at the 150 Mvar rating,
so a saturated STATCOM delivers only Q_RATED / Q_REQ of the support
(synchronising and damping).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

T_DIST = 0.25
T_WIN, T_PRE, T_INF = 0.500, 0.0019, 0.0173
T_DEC = T_DIST + T_WIN + T_PRE + T_INF             # 0.7692 s
TAU_Q = 0.030 / 3.0                                # 95 % in 30 ms
Q_RATED = 150.0

DT = 1e-3
t = np.arange(0.0, 4.0 + DT, DT)

D0 = np.deg2rad(19.0)
PM = 4.0
PMAX0 = PM / np.sin(D0)
P_TARGET = PM / np.sin(np.deg2rad(22.5))
EFF = 0.40
T_REC, TAU_REC = 1.00, 1.00
D_U, D_C = 0.25, 3.0

# case: (Pnet after disturbance, peak reactive-power requirement in Mvar)
CASES = {"T": (2.0, 140.0), "combined": (1.4, 205.0)}


def pnet(tt, p_dist):
    if tt < T_DIST:
        return PMAX0
    s = 0.0 if tt < T_REC else 1 - np.exp(-((tt - T_REC) / TAU_REC) ** 2)
    return p_dist + (P_TARGET - p_dist) * s


def simulate(case, control=True, cap=Q_RATED, t_dec=T_DEC):
    p_dist, q_req = CASES[case]

    def demand_fn(tt):
        return q_req * max(0.0, (P_TARGET - pnet(tt, p_dist))
                           / (P_TARGET - p_dist))

    def rhs(tt, x):
        d, w, q = x
        on = control and tt >= t_dec
        q_cmd = min(demand_fn(tt), cap) if on else 0.0
        pmax = pnet(tt, p_dist) + EFF * (q / q_req) * (P_TARGET - p_dist)
        ramp = 1 - np.exp(-(tt - t_dec) / TAU_Q) if on else 0.0
        damp = D_U + (D_C - D_U) * min(cap / q_req, 1.0) * ramp
        return [w, PM - pmax * np.sin(d) - damp * w, (q_cmd - q) / TAU_Q]

    sol = solve_ivp(rhs, (0, t[-1]), [D0, 0.0, 0.0], t_eval=t, max_step=1e-3,
                    rtol=1e-8, atol=1e-10)
    demand = np.array([demand_fn(x) for x in t])
    return np.rad2deg(sol.y[0]), sol.y[2], demand


def plot(out="statcom_saturation_timing"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    })
    d_nc, _, _ = simulate("combined", control=False)
    d_t, q_t, dem_t = simulate("T")
    d_c, q_c, dem_c = simulate("combined")
    pk_t, pk_c = d_t.max(), d_c.max()
    it, ic = np.argmax(d_t), np.argmax(d_c)
    sat = np.where((q_c >= 0.98 * Q_RATED))[0]
    sat_on, sat_off = t[sat[0]], t[sat[-1]]
    print(f"Case T peak {pk_t:.1f} deg at {t[it]:.2f} s, Qmax {q_t.max():.1f}")
    print(f"Combined peak {pk_c:.1f} deg at {t[ic]:.2f} s, demand max "
          f"{dem_c[t >= T_DEC].max():.0f} Mvar, saturated {sat_on:.3f}-"
          f"{sat_off:.3f} s")
    print(f"NC crosses 180 deg at {t[np.argmax(d_nc > 180)]:.2f} s")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 7.4), dpi=100,
                                   sharex=False)
    green, purple = "#1e9e55", "#8e44ad"

    for ax in (ax1, ax2):
        ax.axvspan(T_DIST, T_DIST + T_WIN, color="0.5", alpha=0.08, lw=0)
        ax.axvline(T_DIST, color="black", ls="--", lw=1.0)
        ax.axvline(T_DEC, color="0.25", lw=0.9)
        ax.grid(True, color="0.85", lw=0.8)
        ax.set_xlim(-0.2, 4.1)

    ax1.plot(t, d_nc, color="#c0392b", ls="--", lw=2.2,
             label="NC — G10 loses synchronism")
    ax1.plot(t, d_t, color=green, ls="-.", lw=2.4,
             label=f"PF unsaturated (Case T alone, $\\delta_{{peak}}$="
                   f"{pk_t:.1f}°)")
    ax1.plot(t, d_c, color=purple, lw=2.8,
             label=f"PF saturated (combined disturbance, $\\delta_{{peak}}$="
                   f"{pk_c:.1f}°)")
    ax1.axhline(180, color="black", ls="--", lw=1.3,
                label="180° stability boundary")
    ax1.axhline(pk_t, color=green, ls=":", lw=0.9)
    ax1.axhline(pk_c, color=purple, ls=":", lw=0.9)
    ax1.annotate(f"$\\delta_{{peak}}$={pk_c:.1f}°, "
                 f"{180 - pk_c:.0f}° margin", xy=(t[ic], pk_c),
                 xytext=(t[ic] + 0.45, pk_c + 30), color=purple, fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color=purple, lw=0.9))
    ax1.annotate(f"$\\delta_{{peak}}$={pk_t:.1f}°", xy=(t[it], pk_t),
                 xytext=(t[it] + 0.6, pk_t - 30), color=green, fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color=green, lw=0.9))
    ax1.text(T_DIST - 0.03, 120, f"Disturbance ($t$={T_DIST:.2f} s)",
             rotation=90, ha="right", va="center", fontsize=8.5)
    ax1.text(T_DEC + 0.03, 125, f"Decision + STATCOM\n($t$={T_DEC:.3f} s)",
             rotation=90, ha="left", va="center", fontsize=8.5)
    ax1.set_ylim(0, 200)
    ax1.set_ylabel("G10 rotor angle $\\delta$ (degrees)", fontsize=12)
    ax1.set_title("(a) G10 Rotor Angle — Effect of STATCOM Saturation",
                  fontsize=11.5)
    ax1.legend(loc="upper right", fontsize=8.6, edgecolor="black",
               fancybox=False, framealpha=1)

    ax2.axvspan(sat_on, sat_off, color="#f5cba7", alpha=0.45, lw=0,
                label=f"Saturation window ({sat_on:.3f}–{sat_off:.2f} s)")
    ax2.plot(t, q_t, color=green, ls="-.", lw=2.4,
             label="Bus 16 STATCOM — Case T alone (no saturation)")
    ax2.plot(t, q_c, color=purple, lw=2.8,
             label="Bus 16 STATCOM — Combined disturbance "
                   "(capped at 150 Mvar)")
    m = (t >= T_DEC) & (dem_c > Q_RATED)
    ax2.plot(t[m], dem_c[m], color=purple, ls=":", lw=1.4,
             label="Combined-disturbance demand (unclipped)")
    ax2.axhline(Q_RATED, color="black", ls="--", lw=1.3,
                label="150 Mvar rated capacity limit")
    ax2.axhline(0.9 * Q_RATED, color="0.4", ls=":", lw=1.0,
                label="90% utilisation threshold")
    ax2.text(T_DIST + T_WIN / 2, 8, "PMU window\n(500 ms)", ha="center",
             va="bottom", fontsize=8.5)
    ax2.set_ylim(0, 200)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("$Q_{FACTS}$ — STATCOM Bus 16 (Mvar)", fontsize=12)
    ax2.set_title("(b) STATCOM Reactive-Power Output — Saturation "
                  "Clamping Visible", fontsize=11.5)
    ax2.legend(loc="upper right", fontsize=8.3, edgecolor="black",
               fancybox=False, framealpha=1)

    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
