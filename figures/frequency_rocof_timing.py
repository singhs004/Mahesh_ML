"""System frequency and ROCOF after the G6 trip, with staged UFLS, from the
shared model freq_model.py."""
import matplotlib.pyplot as plt

import freq_model as fm
import freq_plotlib as fp
import style


def plot(out="frequency_rocof_timing"):
    style.apply()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 8.0), dpi=100)
    fp.event_lines(ax1, ytxt=48.45)
    fp.event_lines(ax2)
    for key, case, tc, c, ls, lw, lab in fp.CASES:
        r = fp.run(key, case, tc)
        mt = fm.metrics(r)
        st = [str(i + 1) for i, v in enumerate(mt["ufls"]) if v is not None]
        ufls = (f"UFLS stage{'s' if len(st) > 1 else ''} {'+'.join(st)}"
                if st else "no UFLS")
        ax1.plot(r["t"], r["f"], color=c, ls=ls, lw=lw,
                 label=f"{lab}: nadir {mt['nadir']:.2f} Hz, {ufls}")
        ax2.plot(r["t"], r["rocof"], color=c, ls=ls, lw=lw,
                 label=f"{lab}: |ROCOF|$_{{max}}$ = {mt['rocof_max']:.2f} Hz/s")
        if key == "NC" and mt["ufls"][0] is not None:
            ax1.annotate(f"UFLS-1 pickup ($t$={mt['ufls'][0]:.2f} s)",
                         xy=(mt["ufls"][0], 49.0),
                         xytext=(mt["ufls"][0] + 2.4, 48.3), fontsize=9,
                         arrowprops=dict(arrowstyle="->", lw=0.9))
    ax1.axhline(49.0, color="black", ls="--", lw=1.2,
                label="UFLS stage-1 (49.0 Hz)")
    ax1.axhline(48.5, color="black", ls=":", lw=1.2,
                label="UFLS stage-2 (48.5 Hz)")
    ax1.set_xlim(-0.3, 10.3)
    ax1.set_ylim(47.9, 50.2)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("System frequency (Hz)", fontsize=12)
    ax1.set_title("(a) System Frequency", fontsize=11.5)
    ax1.grid(True, color="0.85", lw=0.8)
    ax1.legend(loc="lower right", fontsize=7.8, edgecolor="black",
               fancybox=False, framealpha=1)
    ax2.axhline(0, color="black", lw=0.6)
    for s in (1, -1):
        ax2.axhline(2.0 * s, color="black", ls="--", lw=1.2,
                    label="ROCOF relay trip (±2.0 Hz/s)" if s > 0 else None)
        ax2.axhline(1.0 * s, color="0.4", ls=":", lw=1.0,
                    label="ROCOF arming (±1.0 Hz/s)" if s > 0 else None)
    ax2.set_xlim(-0.3, 10.3)
    ax2.set_ylim(-2.4, 1.2)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("ROCOF $df/dt$ (Hz/s)", fontsize=12)
    ax2.set_title("(b) Rate-of-Change-of-Frequency (ROCOF, 100 ms window)",
                  fontsize=11.5)
    ax2.grid(True, color="0.85", lw=0.8)
    ax2.legend(loc="lower right", fontsize=7.8, edgecolor="black",
               fancybox=False, framealpha=1)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


if __name__ == "__main__":
    plot()
