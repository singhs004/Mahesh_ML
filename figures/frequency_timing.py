"""System-frequency response to a load step with the CNN-LSTM + SVC control
chain re-timed to the measured latency budget.

Load step at t = 0.5 s. Decision available 519.2 ms later (t = 1.019 s);
SVC response ~150 ms, so corrective action at t = 1.169 s + T_comm.
Until that instant the CNN-LSTM + SVC trajectory equals the uncontrolled one.
"""
import numpy as np
import matplotlib.pyplot as plt

T_STEP = 0.50
T_WIN, T_PRE, T_INF = 0.500, 0.0019, 0.0173
T_DEC = T_STEP + T_WIN + T_PRE + T_INF             # 1.0192 s
TAU_SVC = 0.150
T_SVC = T_DEC + TAU_SVC                            # 1.1692 s (+T_comm)
F0, UFLS = 50.0, 49.0

DT = 1e-3
t = np.arange(0.0, 5.0 + DT, DT)
rng = np.random.default_rng(3)


def noise(scale=0.008):
    return rng.normal(0.0, scale, t.size)


def response(depth, tau_rec, osc_amp, osc_period, osc_start, osc_decay, ss):
    """Frequency deviation: step drop with exponential recovery plus a
    slower inter-area oscillation that starts at osc_start (s after step)."""
    df = np.zeros_like(t)
    m = t >= T_STEP
    tt = t[m] - T_STEP
    df[m] = -depth * np.exp(-tt / tau_rec) - ss * (1 - np.exp(-tt / 1.0))
    om = tt >= osc_start
    to = tt[om] - osc_start
    osc = np.zeros_like(tt)
    osc[om] = -osc_amp * np.exp(-tt[om] / osc_decay) * \
        (1 - np.cos(2 * np.pi * to / osc_period)) / 2
    df[m] += osc
    return df, osc


df_nc, osc_nc = response(1.70, 0.33, 1.45, 1.65, 0.80, 2.26, 0.08)
df_pi, _ = response(0.90, 0.30, 0.40, 2.20, 0.80, 3.0, 0.04)


def cnn_lstm_svc(t_act, tau=TAU_SVC):
    """Uncontrolled trajectory until t_act; afterwards the SVC output ramps
    in over its response time and suppresses the oscillatory component."""
    df = df_nc.copy()
    m = t >= t_act
    k = 1.0 - np.exp(-(t[m] - t_act) / (tau / 3.0))
    tt = t[m] - T_STEP
    osc = np.interp(t[m], t[t >= T_STEP], osc_nc)
    osc_act = np.interp(t_act, t[t >= T_STEP], osc_nc)
    # oscillation frozen at its value at t_act, then decays under control
    residual = osc_act * np.exp(-(t[m] - t_act) / 0.45)
    ss_nc = 0.08 * (1 - np.exp(-tt / 1.0))
    df[m] = df[m] + k * (residual - osc) + k * ss_nc * 0.9
    return df


def plot(out="frequency_timing"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    })
    tc_late = 0.200
    f_nc = F0 + df_nc
    f_pi = F0 + df_pi
    f_svc0 = F0 + cnn_lstm_svc(T_SVC)
    f_svc2 = F0 + cnn_lstm_svc(T_SVC + tc_late)

    fig, ax = plt.subplots(figsize=(7.5, 4.9), dpi=100)
    ax.axvspan(T_STEP, T_STEP + T_WIN, color="0.5", alpha=0.08, lw=0)
    ax.axhline(UFLS, color="black", lw=1.2, zorder=2)
    ax.text(3.75, UFLS + 0.02, f"UFLS Relay Threshold ({UFLS:.1f} Hz)",
            ha="center", va="bottom", fontsize=9.5)

    me = 310
    series = [
        (f_nc, "#d62728", "o", "-", 100, "No Control"),
        (f_pi, "#2e8b3a", "s", "-", 0, "Conventional PI-AGC"),
        (f_svc0, "#1f6fc5", "^", "-", 200,
         f"CNN-LSTM + SVC ($T_{{comm}}$=0, act. {T_SVC:.3f} s)"),
        (f_svc2, "0.25", "D", "--", 250,
         f"CNN-LSTM + SVC ($T_{{comm}}$={tc_late*1e3:.0f} ms, act. "
         f"{T_SVC + tc_late:.3f} s)"),
    ]
    for y, c, mk, ls, off, lab in series:
        nadir = y[t >= T_STEP].min()
        ax.plot(t, y + noise(), color=c, ls=ls, lw=1.6, marker=mk, ms=6.5,
                markevery=(off, me), zorder=3,
                label=f"{lab} (Nadir: {nadir:.2f} Hz)")
        print(f"{lab:45s} nadir {nadir:.3f} Hz at "
              f"t={t[t >= T_STEP][y[t >= T_STEP].argmin()]:.3f} s")

    def vline(x, text, side="left", y=48.85):
        ax.axvline(x, color="black", lw=1.0, zorder=2)
        dx = -0.02 if side == "left" else 0.02
        ax.text(x + dx, y, text, rotation=90, va="top",
                ha="right" if side == "left" else "left", fontsize=9.5)

    vline(T_STEP, f"Load Step (t={T_STEP:g} s)", y=48.95)
    vline(T_DEC, f"Decision Available (t={T_DEC:.3f} s)")
    vline(T_SVC, f"SVC Action (t={T_SVC:.3f} s)", side="right")
    ax.annotate("", xy=(T_STEP, 50.32), xytext=(T_STEP + T_WIN, 50.32),
                arrowprops=dict(arrowstyle="<->", lw=0.9))
    ax.text(T_STEP + T_WIN / 2, 50.35, "PMU window (500 ms)", ha="center",
            va="bottom", fontsize=8.5,
            bbox=dict(fc="white", ec="none", pad=0.3))

    ax.set_xlim(0, 5)
    ax.set_ylim(47.5, 50.5)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("System Frequency (Hz)", fontsize=12)
    ax.grid(True, color="0.75", lw=0.8)
    ax.legend(loc="lower right", fontsize=8.6, frameon=True,
              edgecolor="black", fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
