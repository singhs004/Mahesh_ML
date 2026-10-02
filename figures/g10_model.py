"""Single shared model for every rotor-angle figure (one case, one set of
numbers).

Event timing (identical in all figures):
    fault onset          T_FAULT  = 0.100 s
    fault cleared        T_CLEAR  = 0.250 s
    decision available   T_DEC    = T_FAULT + 500 + 1.9 + 17.3 ms = 0.619 s
    STATCOM full output  T_DEC + 30 ms  (+ T_comm) = 0.649 s
    SVC full output      T_DEC + 150 ms (+ T_comm) = 0.769 s
The compensator output starts at T_DEC + T_comm and reaches 95 % at the
action instant (first-order lag, tau = response time / 3).

G10 single-machine swing model (rad, s):
    delta'' = PM - Pmax(t) sin(delta) - D(t) delta'
    pre-fault  Pmax = PM / sin(delta0)
    fault      Pmax = K_FAULT * Pmax_pre                    (0.10-0.25 s)
    post-fault Pmax = P_POST[case]  (< PM: no equilibrium without support)
    support    Pmax += (P_TGT - P_POST) * s(t) * u(t)
where u(t) is the normalised compensator output (transient boost that
settles to U_SS) and s(t) <= 1 is the delivered fraction when the STATCOM
saturates. Reactive power Q(t) = Q_REQ * u(t), clipped at Q_RATED.
G1/G4/G7 are the stable machines of the same event.
"""
import json
import numpy as np
from scipy.integrate import solve_ivp

T_FAULT, T_CLEAR = 0.10, 0.25
T_WIN, T_PRE, T_INF = 0.500, 0.0019, 0.0173
T_DEC = T_FAULT + T_WIN + T_PRE + T_INF            # 0.6192 s
TAU = {"STATCOM": 0.030, "SVC": 0.150}
T_ACT = {k: T_DEC + v for k, v in TAU.items()}     # 0.6492 / 0.7692 s

DT = 1e-3
T_END = 4.0
t = np.round(np.arange(0.0, T_END + DT / 2, DT), 6)

DELTA0 = 19.0                                      # deg, pre-fault G10
PM = 7.0
P_PRE = PM / np.sin(np.deg2rad(DELTA0))
K_FAULT = 0.0                                      # bolted fault close to G10
P_POST = {"T": 5.0, "CC": 5.6, "combined": 4.4}
P_TGT = PM / np.sin(np.deg2rad(26.0))
U_SS, T_HOLD, TAU_REL = 0.60, 0.92, 0.20
D_U, D_C = 0.25, 2.5
Q_RATED = 150.0
Q_REQ = {"T": 148.2, "combined": 168.4}     # paper Tables 12 and 31


def u_profile(tt, t_start, tau):
    """Normalised compensator output (0 before t_start)."""
    if tt < t_start:
        return 0.0, 0.0
    ramp = 1.0 - np.exp(-(tt - t_start) / (tau / 3.0))
    rel = 1.0 if tt < T_HOLD else \
        U_SS + (1 - U_SS) * np.exp(-(tt - T_HOLD) / TAU_REL)
    return ramp * rel, ramp


def g10(case="T", device=None, t_comm=0.0, cap=Q_RATED):
    """G10 rotor angle (deg) and STATCOM/SVC output (Mvar).
    case: 'T' (G10 contingency), 'CC' (conventional control), 'combined'.
    device: None (no corrective control), 'STATCOM' or 'SVC'."""
    p_post = P_POST[case]
    q_req = Q_REQ.get(case, Q_REQ["T"])
    t_start = T_DEC + t_comm
    tau = TAU.get(device, 0.03)

    def support(tt):
        if device is None:
            return 0.0, 0.0, 0.0
        u, ramp = u_profile(tt, t_start, tau)
        q = min(q_req * u, cap)
        s = q / (q_req * u) if u > 0 else 1.0
        return u * s, ramp * min(cap / q_req, 1.0), q

    def rhs(tt, x):
        if tt < T_FAULT:
            pmax, d = P_PRE, D_U
        elif tt < T_CLEAR:
            pmax, d = K_FAULT * P_PRE, D_U
        else:
            eff, damp, _ = support(tt)
            pmax = p_post + (P_TGT - p_post) * eff
            d = D_U + (D_C - D_U) * damp
        return [x[1], PM - pmax * np.sin(x[0]) - d * x[1]]

    sol = solve_ivp(rhs, (0, T_END), [np.deg2rad(DELTA0), 0.0], t_eval=t,
                    max_step=1e-3, rtol=1e-8, atol=1e-10)
    q = np.array([support(x)[2] for x in t])
    return np.rad2deg(sol.y[0]), q


# stable machines: (pre-fault angle, natural frequency Hz, fault-on Pmax
# factor, post-fault Pmax factor, damping); remote from the fault, so they
# accelerate less than G10 during 0.10-0.25 s
GENS = {"G1": (20.0, 0.70, 0.55, 0.95, 0.9),
        "G4": (25.0, 0.65, 0.50, 0.93, 0.7),
        "G7": (28.0, 0.62, 0.45, 0.92, 0.5)}
H_GEN = {"G1": 5.5, "G4": 4.8, "G7": 4.5, "G10": 3.5}


def stable_gens():
    """G1, G4, G7 absolute rotor angles (deg) from the same swing equation;
    the angle leaves its pre-fault value with zero slope (speed deviation
    is zero at fault onset)."""
    out = {}
    for name, (d0, fn, kf, kp, damp) in GENS.items():
        pmax = (2 * np.pi * fn) ** 2 / np.cos(np.deg2rad(d0))
        pm = pmax * np.sin(np.deg2rad(d0))

        def rhs(tt, x, pmax=pmax, pm=pm, kf=kf, kp=kp, damp=damp):
            k = 1.0 if tt < T_FAULT else (kf if tt < T_CLEAR else kp)
            return [x[1], pm - k * pmax * np.sin(x[0]) - damp * x[1]]

        sol = solve_ivp(rhs, (0, T_END), [np.deg2rad(d0), 0.0], t_eval=t,
                        max_step=1e-3, rtol=1e-8, atol=1e-10)
        out[name] = np.rad2deg(sol.y[0])
    return out


def cross_time(y, level=180.0):
    i = np.argmax(y > level)
    return float(t[i]) if y[i] > level else None


def critical_tcomm(device, lo=0.0, hi=0.6, tol=1e-3):
    if g10("T", device, hi)[0].max() < 180:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if g10("T", device, mid)[0].max() < 180:
            lo = mid
        else:
            hi = mid
    return lo


def summary(path="g10_results.json"):
    """Compute every G10 number used in the figures and tables."""
    nc, _ = g10("T")
    cc, _ = g10("CC")
    res = {"delta0": DELTA0, "t_dec": T_DEC, "t_act": T_ACT,
           "nc_cross": cross_time(nc), "cc_cross": cross_time(cc),
           "sweep": [], "gens": {}}
    for name, y in stable_gens().items():
        res["gens"][name] = {"peak": float(y.max()),
                             "peak_dev": float(y.max() - y[0]),
                             "at_clear_dev": float(y[t == T_CLEAR][0] - y[0])}
    res["g10_at_clear_dev"] = float(nc[t == T_CLEAR][0] - DELTA0)
    for tc in [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
        row = {"tcomm_ms": tc}
        for dev in ("STATCOM", "SVC"):
            y, _ = g10("T", dev, tc / 1e3)
            stable = bool(y.max() < 180)
            row[dev] = {"act": T_ACT[dev] + tc / 1e3,
                        "peak": float(y.max()) if stable else None,
                        "peak_dev": float(y.max() - DELTA0) if stable else None,
                        "stable": stable}
        res["sweep"].append(row)
    res["crit"] = {d: critical_tcomm(d) for d in ("STATCOM", "SVC")}
    yc, qc = g10("combined", "STATCOM")
    ycn, _ = g10("combined")
    yt, qt = g10("T", "STATCOM")
    sat = np.where(qc >= 0.98 * Q_RATED)[0]
    res["sat"] = {"T_peak": float(yt.max()), "T_qmax": float(qt.max()),
                  "comb_peak": float(yc.max()),
                  "comb_demand": Q_REQ["combined"],
                  "window": [float(t[sat[0]]), float(t[sat[-1]])]
                  if sat.size else None,
                  "comb_nc_cross": cross_time(ycn)}
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    r = summary()
    print(json.dumps({k: v for k, v in r.items() if k != "sweep"}, indent=1))
    for row in r["sweep"]:
        print(row["tcomm_ms"], row["STATCOM"]["peak"], row["SVC"]["peak"])
