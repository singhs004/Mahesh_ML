"""Bus 44 voltage-collapse scenario (NC / CC / PF) with the CNN-LSTM + SVC
action re-timed to the measured latency budget.

Line trip at t = 0.5 s. Decision available 519.2 ms later (t = 1.019 s);
SVC response ~150 ms, so reactive support starts at t = 1.169 s + T_comm.
Until that instant the PF trajectory equals the no-control (NC) one.
"""
import numpy as np
import matplotlib.pyplot as plt

T_TRIP = 0.50
T_WIN, T_PRE, T_INF = 0.500, 0.0019, 0.0173
T_DEC = T_TRIP + T_WIN + T_PRE + T_INF             # 1.0192 s
TAU_SVC = 0.150
T_SVC = T_DEC + TAU_SVC                            # 1.1692 s (+T_comm)
V0, V_ANSI, V_UV = 0.942, 0.90, 0.60                # V0 from paper Table 20
V_NC_MIN, V_CC_MIN, V_CC_END, V_PF_END = 0.621, 0.782, 0.86, 0.938  # Table 20

DT = 1e-3
t = np.arange(0.0, 10.0 + DT, DT)


def v_nc_fn(tt):
    """No control: smooth, monotonic voltage collapse towards the Table 20
    minimum (no step changes; slope is zero at the line trip)."""
    tt = np.asarray(tt, dtype=float)
    tau = np.clip(tt - T_TRIP, 0, None)
    a = 1.0
    return V_NC_MIN + (V0 - V_NC_MIN) * (1 + tau / a) * np.exp(-tau / a)


def v_cc_fn(tt):
    """Conventional control: same initial decline as NC but shallower
    (always above NC), smooth recovery towards the Table 20 value at 10 s."""
    tt = np.asarray(tt, dtype=float)
    tau = np.clip(tt - T_TRIP, 0, None)
    a = 1.0
    dip = V0 - 0.700
    v = V0 - dip * (1 - (1 + tau / a) * np.exp(-tau / a))
    rec = 0.1623 / (1 + np.exp(-(tt - 5.0) / 1.2))
    return v + rec


v_nc = v_nc_fn(t)
v_cc = v_cc_fn(t)


def pf(t_act, tau=TAU_SVC, v_end=V_PF_END, tau_rec=0.9):
    """NC trajectory until t_act; then SVC support ramps in over its
    response time and drives the voltage to the post-disturbance level."""
    v = v_nc.copy()
    m = t >= t_act
    v_act = v_nc_fn(t_act)
    target = v_end - (v_end - v_act) * np.exp(-(t[m] - t_act) / tau_rec)
    k = 1.0 - np.exp(-(t[m] - t_act) / (tau / 3.0))
    v[m] = v_nc[m] + k * (target - v_nc[m])
    return v


def plot(out="bus44_collapse_timing"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    })
    tc_late = 0.200
    v_pf = pf(T_SVC)
    v_pf2 = pf(T_SVC + tc_late)

    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=100)
    ax.axvspan(T_TRIP, T_TRIP + T_WIN, color="0.5", alpha=0.10, lw=0)
    ax.fill_between(t, V_UV, V_ANSI, where=t >= 0.85, color="#e74c3c",
                    alpha=0.08, lw=0)

    nadir = {}
    for v, c, ls, lw, lab in [
        (v_nc, "#c0392b", "--", 2.2, "NC — Voltage collapse"),
        (v_cc, "#e67e22", "-.", 2.2,
         f"CC — Partial recovery ({v_cc[-1]:.2f} p.u. at 10 s)"),
        (v_pf, "#27a050", "-", 2.8,
         f"PF — Recovery (CNN-LSTM+SVC, $T_{{comm}}$=0, act. {T_SVC:.3f} s)"),
        (v_pf2, "0.3", ":", 2.2,
         f"PF — Recovery (CNN-LSTM+SVC, $T_{{comm}}$={tc_late*1e3:.0f} ms, "
         f"act. {T_SVC + tc_late:.3f} s)"),
    ]:
        ax.plot(t, v, color=c, ls=ls, lw=lw, label=lab, zorder=3)
        m = t >= T_TRIP
        i = np.argmin(v[m])
        nadir[lab[:2] + str(lw)] = (v[m][i], t[m][i])
        print(f"{lab:72s} nadir {v[m][i]:.3f} p.u. at {t[m][i]:.2f} s; "
              f"below 0.90 for {((v < V_ANSI) & m).sum() * DT:.2f} s")

    ax.axhline(V_ANSI, color="black", ls="--", lw=1.3,
               label=f"ANSI C84.1 lower limit ({V_ANSI:.2f} p.u.)")
    ax.axhline(V_UV, color="black", ls=":", lw=1.3,
               label=f"UV relay trip ({V_UV:.2f} p.u.)")
    ax.axvline(T_TRIP, color="black", ls="--", lw=1.0)
    for x in (T_DEC, T_SVC):
        ax.axvline(x, color="0.25", lw=0.9)
    ax.text(T_DEC - 0.04, 0.70, f"Decision available ($t$={T_DEC:.3f} s)",
            rotation=90, ha="right", va="center", fontsize=8.5)
    ax.text(T_SVC + 0.04, 0.70, f"SVC action ($t$={T_SVC:.3f} s)",
            rotation=90, ha="left", va="center", fontsize=8.5)
    ax.text(T_TRIP - 0.04, 0.70, f"Line trip ($t$={T_TRIP:.2f} s)",
            rotation=90, ha="right", va="center", fontsize=8.5)
    ax.annotate("", xy=(T_TRIP, 0.52), xytext=(T_TRIP + T_WIN, 0.52),
                arrowprops=dict(arrowstyle="<->", lw=0.9))
    ax.text(T_SVC + 0.2, 0.52, "PMU window (500 ms)", ha="left",
            va="center", fontsize=8.5)

    vn, tn = nadir["PF2.8"]
    ax.annotate(f"Nadir {vn:.3f} p.u.\n(PF, $t$={tn:.2f} s)", xy=(tn, vn),
                xytext=(2.3, 0.865), color="#1e8449", fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1.0))

    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 0.96)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Bus 44 voltage magnitude (p.u.)", fontsize=12)
    ax.grid(True, color="0.85", lw=0.8)
    ax.legend(loc="lower right", fontsize=8.2, edgecolor="black",
              fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
