"""Generator and BESS active power after the G6 trip, with the BESS dispatch
re-timed to the CNN-LSTM latency budget.

G6 trips at t = 0.5 s. The CNN-LSTM decision is available 519.2 ms later
(t = 1.019 s); the BESS units are dispatched at that instant (+ T_comm) and
ramp at 250 MW/s to their set-points (Bus 3: 50 MW, Bus 29: 78 MW). They
hold until AGC takes over (t = 6.55 s) and then decay.
"""
import numpy as np
import matplotlib.pyplot as plt

T_TRIP = 0.50
T_WIN, T_PRE, T_INF = 0.500, 0.0019, 0.0173
T_DEC = T_TRIP + T_WIN + T_PRE + T_INF             # 1.0192 s
RAMP = 250.0                                       # MW/s
T_HANDOVER, TAU_DECAY = 6.55, 1.3
FIRST_SWING = (0.50, 0.70)

DT = 1e-3
t = np.arange(0.0, 10.0 + DT, DT)


def gen(base, delta, tau, tau_agc=None, agc=0.0):
    p = np.full_like(t, base)
    m = t >= T_TRIP
    tt = t[m] - T_TRIP
    p[m] += delta * (1 - np.exp(-tt / tau))
    if tau_agc:
        p[m] += agc * (1 - np.exp(-tt / tau_agc))
    return p


def bess(p_set, t_dispatch):
    p = np.zeros_like(t)
    m = t >= t_dispatch
    p[m] = np.minimum(RAMP * (t[m] - t_dispatch), p_set)
    d = t >= T_HANDOVER
    p[d] = p[d] * np.exp(-(t[d] - T_HANDOVER) / TAU_DECAY)
    return p


def plot(out="bess_power_timing", t_comm=0.0):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    })
    g1 = gen(250, 14, 0.9)
    g3 = gen(263, 10, 0.9, 2.5, 14)
    g9 = gen(185, 18, 0.55)
    g6 = np.where(t < T_TRIP, 225.0, 0.0)
    t_disp = T_DEC + t_comm
    b3, b29 = bess(50, t_disp), bess(78, t_disp)

    fs = (t >= FIRST_SWING[0]) & (t <= FIRST_SWING[1])
    e_fs = (b3[fs] + b29[fs]).sum() * DT
    full3, full29 = t_disp + 50 / RAMP, t_disp + 78 / RAMP
    print(f"dispatch {t_disp:.3f} s; Bus 3 full at {full3:.3f} s, "
          f"Bus 29 full at {full29:.3f} s; energy in first-swing window "
          f"{e_fs:.2f} MJ; total BESS at 0.70 s = "
          f"{b3[fs][-1] + b29[fs][-1]:.0f} MW")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 7.6), dpi=100)
    for ax in (ax1, ax2):
        ax.axvline(T_TRIP, color="black", ls="--", lw=1.0)
        ax.axvline(T_DEC, color="0.3", lw=0.9)
        ax.axvspan(T_TRIP, T_TRIP + T_WIN, color="0.5", alpha=0.08, lw=0)
        ax.grid(True, color="0.85", lw=0.8)
        ax.set_xlim(-0.5, 10.5)

    ax1.plot(t, g1, color="#1f6fb4", lw=2.2,
             label="G1 — governor response (base 250 MW)")
    ax1.plot(t, g3, color="#228b3b", lw=2.2, ls="--",
             label="G3 — governor + AGC (base 263 MW)")
    ax1.plot(t, g9, color="#8b6508", lw=2.2, ls="-.",
             label="G9 — governor + AGC (base 185 MW)")
    ax1.plot(t, g6, color="#d62728", lw=2.2, ls=":",
             label=f"G6 — trip at $t$={T_TRIP:.2f} s (−225 MW)")
    ax1.set_ylim(-15, 310)
    ax1.set_ylabel("Active power $P$ (MW)", fontsize=12)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_title("(a) Synchronous Generator Active Power", fontsize=11.5)
    ax1.legend(loc="center right", bbox_to_anchor=(0.995, 0.45), fontsize=8.8,
               edgecolor="black", fancybox=False, framealpha=1)

    ax2.axvspan(*FIRST_SWING, color="#f5b041", alpha=0.35, lw=0)
    ax2.fill_between(t, 0, b29, color="#8e44ad", alpha=0.15, lw=0)
    ax2.fill_between(t, 0, b3, color="#1e9e55", alpha=0.15, lw=0)
    ax2.plot(t, b3, color="#1e9e55", lw=2.6,
             label=f"BESS at Bus 3 (+50 MW, ramp {RAMP:.0f} MW/s)")
    ax2.plot(t, b29, color="#8e44ad", lw=2.6, ls="--",
             label=f"BESS at Bus 29 (+78 MW, ramp {RAMP:.0f} MW/s)")
    ax2.annotate("First-swing window\n(0.50–0.70 s):\n"
                 f"{b3[fs][-1] + b29[fs][-1]:.0f} MW delivered",
                 xy=(0.6, 100), xytext=(1.9, 100), color="#b9770e",
                 fontsize=9.5, va="center",
                 arrowprops=dict(arrowstyle="->", color="#b9770e", lw=0.9))
    ax2.text(T_DEC - 0.06, 4, f"CNN-LSTM dispatch ($t$={t_disp:.3f} s)",
             rotation=90, ha="right", va="bottom", fontsize=8.5,
             bbox=dict(fc="white", ec="none", pad=0.3))
    ax2.text(T_TRIP - 0.07, 4, f"G6 trip ($t$={T_TRIP:.2f} s)", rotation=90,
             ha="right", va="bottom", fontsize=8.5)
    ax2.annotate(f"128 MW reached at $t$={full29:.3f} s", xy=(full29, 78),
                 xytext=(2.2, 66), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=0.9))
    ax2.set_ylim(0, 115)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("BESS active power (MW)", fontsize=12)
    ax2.set_title("(b) Battery Energy Storage System (BESS) Fast Active-Power "
                  "Response", fontsize=11.5)
    ax2.legend(loc="upper right", fontsize=8.8, edgecolor="black",
               fancybox=False, framealpha=1)

    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
