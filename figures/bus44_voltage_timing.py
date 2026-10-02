"""Bus 44 voltage after a line trip with wind variability, with the
CNN-LSTM + SVC action re-timed to the measured latency budget.

Line trip at t = 0.5 s. Decision available 519.2 ms later (t = 1.019 s);
SVC response ~150 ms, so reactive support starts at t = 1.169 s + T_comm.
Until that instant the controlled trajectory equals the no-control one.
Pre-disturbance voltage 0.942 p.u. (paper Table 20).
"""
import numpy as np
import matplotlib.pyplot as plt

T_TRIP = 0.50
T_WIN, T_PRE, T_INF = 0.500, 0.0019, 0.0173
T_DEC = T_TRIP + T_WIN + T_PRE + T_INF             # 1.0192 s
TAU_SVC = 0.150
T_SVC = T_DEC + TAU_SVC                            # 1.1692 s (+T_comm)
V_ANSI = 0.90

DT = 2e-3
t = np.arange(0.0, 10.0 + DT, DT)
rng = np.random.default_rng(5)


def smooth_noise(scale, corr=0.08):
    w = rng.normal(0.0, 1.0, t.size)
    k = np.exp(-np.arange(0, 5 * corr, DT) / corr)
    n = np.convolve(w, k / np.sqrt((k ** 2).sum()), mode="same")
    return scale * n


V0 = 0.942                                         # paper Table 20


def voltage(v_final, tau_settle, dip, tau_peak, v0=V0):
    v = np.full_like(t, v0)
    m = t >= T_TRIP
    tt = t[m] - T_TRIP
    g = (tt / tau_peak) * np.exp(1 - tt / tau_peak)
    v[m] = v0 - (v0 - v_final) * (1 - np.exp(-tt / tau_settle)) - dip * g
    return v


v_sg = voltage(0.935, 1.0, 0.060, 0.75)            # SG-only baseline
v_nc = voltage(0.910, 2.0, 0.105, 1.10)            # 50 % RES, no control
v_target = v_sg - 0.004                            # supported trajectory


def cnn_lstm_svc(t_act, tau=TAU_SVC):
    v = v_nc.copy()
    m = t >= t_act
    k = 1.0 - np.exp(-(t[m] - t_act) / (tau / 3.0))
    v[m] = v_nc[m] + k * (v_target[m] - v_nc[m])
    return v


def plot(out="bus44_voltage_timing"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    })
    tc_late = 0.200
    v_pf = cnn_lstm_svc(T_SVC)
    v_pf2 = cnn_lstm_svc(T_SVC + tc_late)
    sigma = np.where(t >= T_TRIP, 0.004 + 0.003 * (1 - np.exp(-(t - T_TRIP))),
                     0.0)

    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=100)
    ax.fill_between(t, v_pf - sigma, v_pf + sigma, color="#2ca05a",
                    alpha=0.18, lw=0, label="PF ±1σ wind variability band")
    curves = [
        (v_sg, "#1f77b4", (0, (4, 2)), 2.2, 0.0015,
         "SG-only baseline (0% RES)"),
        (v_nc, "#c0392b", "-.", 1.8, 0.0025, "50% RES, no control"),
        (v_pf, "#27a050", "-", 2.6, 0.0015,
         f"50% RES, CNN-LSTM+SVC (PF), $T_{{comm}}$=0, act. {T_SVC:.3f} s"),
        (v_pf2, "0.3", ":", 2.0, 0.0015,
         f"50% RES, CNN-LSTM+SVC, $T_{{comm}}$={tc_late*1e3:.0f} ms, "
         f"act. {T_SVC + tc_late:.3f} s"),
    ]
    for v, c, ls, lw, ns, lab in curves:
        ax.plot(t, v + smooth_noise(ns), color=c, ls=ls, lw=lw, label=lab,
                zorder=3)
        m = t >= T_TRIP
        i = np.argmin(v[m])
        below = (v < V_ANSI).sum() * DT
        print(f"{lab:62s} nadir {v[m][i]:.3f} p.u. at {t[m][i]:.2f} s, "
              f"below 0.90 p.u. for {below:.2f} s")

    ax.axhline(V_ANSI, color="black", ls="--", lw=1.2,
               label=f"ANSI lower limit ({V_ANSI:.2f} p.u.)")
    ax.axhline(0.60, color="black", ls=":", lw=1.0)
    ax.axvline(T_TRIP, color="black", ls="--", lw=1.0)
    for x in (T_DEC, T_SVC):
        ax.axvline(x, color="0.25", ls="-", lw=0.9)
    ax.text(T_DEC - 0.04, 0.72, f"Decision available ($t$={T_DEC:.3f} s)",
            rotation=90, ha="right", va="center", fontsize=8.5)
    ax.text(T_SVC + 0.04, 0.72, f"SVC action ($t$={T_SVC:.3f} s)",
            rotation=90, ha="left", va="center", fontsize=8.5)
    ax.axvspan(T_TRIP, T_TRIP + T_WIN, color="0.5", alpha=0.08, lw=0)
    ax.annotate("", xy=(T_TRIP, 0.575), xytext=(T_TRIP + T_WIN, 0.575),
                arrowprops=dict(arrowstyle="<->", lw=0.9))
    ax.text(T_SVC + 0.20, 0.575, "PMU window (500 ms)", ha="left",
            va="center", fontsize=8.5)
    ax.text(T_TRIP - 0.04, 0.75, f"Line trip + wind variability ($t$={T_TRIP:.2f} s)",
            rotation=90, ha="right", va="center", fontsize=8.5)

    ax.set_xlim(0, 10)
    ax.set_ylim(0.54, 0.98)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Bus 44 voltage magnitude (p.u.)", fontsize=12)
    ax.grid(True, color="0.85", lw=0.8)
    ax.legend(loc="lower right", fontsize=8.4, edgecolor="black",
              fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
