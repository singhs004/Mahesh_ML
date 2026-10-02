"""System frequency and ROCOF after a generator trip + load step, with the
CNN-LSTM + SVC action re-timed to the measured latency budget.

Disturbance at t = 0.5 s. Decision available 519.2 ms later (t = 1.019 s);
SVC response ~150 ms, so PF support starts at t = 1.169 s + T_comm.

Single-area system-frequency-response model (per unit on system base):
    2H dx/dt   = Pm + P_ctrl + P_shed - dP_L - D x          (x = df / f0)
    Tg dPm/dt  = -x / R - Pm                                 (governor)
UFLS: stage 1 sheds S1 when f < 49.0 Hz, stage 2 sheds S2 when f < 48.5 Hz
after a T_RELAY pickup + breaker delay (applied identically to every case).
CC / PI-AGC: P_ctrl = -(Kp x + Ki int x), active from the disturbance.
PF: P_ctrl = SVC-enabled frequency support, zero until the action instant,
then ramped in with a first-order lag (95 % in 150 ms).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

F0 = 50.0
T_DIST = 0.50
T_WIN, T_PRE, T_INF = 0.500, 0.0019, 0.0173
T_DEC = T_DIST + T_WIN + T_PRE + T_INF             # 1.0192 s
TAU_SVC = 0.150
T_SVC = T_DEC + TAU_SVC                            # 1.1692 s (+T_comm)
UFLS1, UFLS2 = 49.0, 48.5
S1, S2 = 0.12, 0.15
T_RELAY = 0.25                                     # UFLS pickup + breaker delay
ROCOF_TRIP, ROCOF_ARM = 2.0, 1.0

H, D, R, TG = 3.0, 1.0, 1.0, 5.0
DPL = 0.30
KP_CC, KI_CC = 1.5, 0.3
P_SVC, K_FFR = 0.0, 4.0

DT = 1e-3
t = np.arange(0.0, 10.0 + DT, DT)


def simulate(case, t_act=T_SVC):
    """Return frequency (Hz), ROCOF (Hz/s) and UFLS trip times."""
    x, pm, ix, ps = 0.0, 0.0, 0.0, 0.0
    shed1 = shed2 = None
    f = np.empty_like(t)
    for i, tt in enumerate(t):
        f[i] = F0 * (1 + x)
        if tt < T_DIST:
            continue
        fh = F0 * (1 + x)
        if shed1 is None and fh < UFLS1:
            shed1 = tt
        if shed2 is None and fh < UFLS2:
            shed2 = tt
        p_target = (S1 if shed1 is not None and tt >= shed1 + T_RELAY
                  else 0.0) + \
                 (S2 if shed2 is not None and tt >= shed2 + T_RELAY else 0.0)
        ps += (p_target - ps) / 0.08 * DT          # feeder/load disconnection
        p_shed = ps
        p_ctrl = 0.0
        if case == "CC":
            p_ctrl = -(KP_CC * x + KI_CC * ix)
        elif case == "PF" and tt >= t_act:
            k = 1 - np.exp(-(tt - t_act) / (TAU_SVC / 3))
            p_ctrl = k * (P_SVC - K_FFR * x)
        dx = (pm + p_ctrl + p_shed - DPL - D * x) / (2 * H)
        dpm = (-x / R - pm) / TG
        x += dx * DT
        pm += dpm * DT
        ix += x * DT
    rocof = np.gradient(f, DT)
    # 100 ms moving-average window, as used by ROCOF relays
    w = int(0.1 / DT)
    rocof = np.convolve(rocof, np.ones(w) / w, mode="full")[:t.size]  # causal
    return f, rocof, shed1, shed2


def plot(out="frequency_rocof_timing"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    })
    tc_late = 0.200
    runs = [
        ("NC", T_SVC, "#c0392b", "--", 2.2, "NC"),
        ("CC", T_SVC, "#e67e22", "-.", 2.2, "CC/PI-AGC"),
        ("PF", T_SVC, "#1e9e55", "-", 2.8,
         f"PF ($T_{{comm}}$=0, act. {T_SVC:.3f} s)"),
        ("PF", T_SVC + tc_late, "0.35", ":", 2.2,
         f"PF ($T_{{comm}}$={tc_late*1e3:.0f} ms, act. {T_SVC + tc_late:.3f} s)"),
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 8.0), dpi=100)
    for ax in (ax1, ax2):
        ax.axvspan(T_DIST, T_DIST + T_WIN, color="0.5", alpha=0.08, lw=0)
        ax.axvline(T_DIST, color="black", ls="--", lw=1.0)
        ax.axvline(T_DEC, color="0.3", lw=0.9)
        ax.axvline(T_SVC, color="0.3", lw=0.9)
        ax.grid(True, color="0.85", lw=0.8)
        ax.set_xlim(-0.5, 10.3)

    nc_trips = None
    for case, ta, c, ls, lw, name in runs:
        f, r, s1, s2 = simulate(case, ta)
        m = t >= T_DIST
        nadir = f[m].min()
        rmax = np.abs(r[m]).max()
        stages = [s for s, v in (("1", s1), ("2", s2)) if v is not None]
        ufls = (f"UFLS stage{'s' if len(stages) > 1 else ''} "
                f"{'+'.join(stages)}") if stages else "UFLS not triggered"
        ax1.plot(t, f, color=c, ls=ls, lw=lw,
                 label=f"{name} (nadir {nadir:.2f} Hz, {ufls})")
        ax2.plot(t, r, color=c, ls=ls, lw=lw,
                 label=f"{name} (|ROCOF|$_{{max}}$={rmax:.2f} Hz/s)")
        print(f"{name:32s} nadir {nadir:.2f} Hz at {t[m][f[m].argmin()]:.2f} s; "
              f"UFLS1 {s1}; UFLS2 {s2}; |ROCOF|max {rmax:.2f} Hz/s")
        if case == "NC":
            nc_trips = (s1, s2)

    ax1.axhline(UFLS1, color="black", ls="--", lw=1.2,
                label=f"UFLS stage-1 ({UFLS1:.1f} Hz)")
    ax1.axhline(UFLS2, color="black", ls=":", lw=1.2,
                label=f"UFLS stage-2 ({UFLS2:.1f} Hz)")
    for ts, fv, lab, dy in [(nc_trips[0], UFLS1, "UFLS-1", 0.35),
                            (nc_trips[1], UFLS2, "UFLS-2", 0.30)]:
        ax1.annotate(f"{lab} ($t$={ts:.2f} s)", xy=(ts, fv),
                     xytext=(ts + 1.6, fv + dy), fontsize=9,
                     arrowprops=dict(arrowstyle="->", lw=0.9))
    ax1.text(T_DEC - 0.08, 46.88, f"Decision ($t$={T_DEC:.3f} s)",
             rotation=90, ha="right", va="bottom", fontsize=8.5)
    ax1.text(T_SVC + 0.08, 46.88, f"SVC action ($t$={T_SVC:.3f} s)",
             rotation=90, ha="left", va="bottom", fontsize=8.5)
    ax1.set_ylim(46.8, 50.3)
    ax1.set_ylabel("System frequency (Hz)", fontsize=12)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_title("(a) System Frequency", fontsize=11.5)
    ax1.legend(loc="lower right", fontsize=8.0, edgecolor="black",
               fancybox=False, framealpha=1)

    ax2.axhline(0, color="black", lw=0.6)
    for s in (1, -1):
        ax2.axhline(s * ROCOF_TRIP, color="black", ls="--", lw=1.2,
                    label=f"ROCOF relay trip (±{ROCOF_TRIP:.1f} Hz/s)"
                    if s == 1 else None)
        ax2.axhline(s * ROCOF_ARM, color="0.4", ls=":", lw=1.0,
                    label=f"ROCOF arming (±{ROCOF_ARM:.1f} Hz/s)"
                    if s == 1 else None)
    ax2.set_ylim(-3.5, 2.3)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("ROCOF $df/dt$ (Hz/s)", fontsize=12)
    ax2.set_title("(b) Rate-of-Change-of-Frequency (ROCOF)", fontsize=11.5)
    ax2.legend(loc="lower right", fontsize=8.0, edgecolor="black",
               fancybox=False, framealpha=1)

    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
