"""Helpers shared by the frequency figures."""
import freq_model as fm

CASES = [  # key, case, T_comm, colour, linestyle, width, label
    ("NC", "NC", 0.0, "#c0392b", "--", 2.2, "No control (governor only)"),
    ("CC", "CC", 0.0, "#e67e22", "-.", 2.2, "PI-AGC (governor + AGC)"),
    ("PF", "PF", 0.0, "#1e9e55", "-", 2.8,
     "CNN-LSTM + BESS ($T_{comm}$=0)"),
    ("PF200", "PF", 0.2, "0.35", ":", 2.2,
     "CNN-LSTM + BESS ($T_{comm}$=200 ms)"),
]
_cache = {}


def run(key, case, tc, system="IEEE 39-Bus", t_end=10.0):
    k = (key, system, t_end)
    if k not in _cache:
        _cache[k] = fm.simulate(case, tc, system, t_end)
    return _cache[k]


def event_lines(ax, ytxt=None, size=8.3):
    ax.axvline(fm.T_EVT, color="black", ls="--", lw=1.0)
    ax.axvspan(fm.T_EVT, fm.T_EVT + fm.T_WIN, color="0.5", alpha=0.08, lw=0)
    ax.axvline(fm.T_DEC, color="0.3", lw=0.9)
    if ytxt is not None:
        ax.text(fm.T_EVT - 0.05, ytxt, f"G6 trip ({fm.T_EVT:.2f} s)",
                rotation=90, ha="right", va="center", fontsize=size,
                bbox=dict(fc="white", ec="none", pad=0.2))
        ax.text(fm.T_DEC + 0.05, ytxt,
                f"BESS dispatch ({fm.T_DEC:.3f} s)",
                rotation=90, ha="left", va="center", fontsize=size,
                bbox=dict(fc="white", ec="none", pad=0.2))
