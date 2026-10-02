"""Shared frequency model for the G6-trip event (Figures: system frequency,
frequency deviation, frequency + ROCOF, generator/BESS power).

Parameters from the paper:
  Table 3  governor R = 0.05 (gain 20), Tg = 0.2 s, Tt = 0.5 s,
           reheat Tr = 7.0 s; AGC Kp = 2.0, Ki = 1.0, T = 5 s,
           ramp limit 0.1 p.u./min
  Table 4  generator ratings, set-points and headroom (G6 = 650 MW)
  Table 8  IEEE 39-bus High-RES: H_sys = 1.72 s, total generation 6298 MW
  Table 35 decision available 519.2 ms after onset
BESS (Bus 3: 50 MW, Bus 29: 78 MW, ramp 250 MW/s) is dispatched by the
CNN-LSTM decision at T_DEC + T_comm. Frequency is controlled by governors,
AGC, UFLS and BESS (no FACTS device in these cases).
"""
import json
import numpy as np

F0 = 50.0
T_EVT = 0.50
T_WIN, T_PRE, T_INF = 0.500, 0.0019, 0.0173
T_DEC = T_EVT + T_WIN + T_PRE + T_INF              # 1.0192 s

S_BASE = 6298.0                                    # MW (Table 8)
SYSTEMS = {"IEEE 9-Bus": 3.01, "IEEE 39-Bus": 1.72, "IEEE 118-Bus": 2.10}
D_LOAD = 1.0                                       # p.u. MW / p.u. Hz
P_TRIP = 650.0                                     # G6 (Table 4)

# Table 4: (rating MW, set-point MW, headroom MW); G6 trips
GEN = {"G1": (1040, 250, 790), "G2": (646, 678, 0), "G3": (725, 650, 75),
       "G4": (652, 632, 20), "G5": (508, 508, 0), "G7": (580, 560, 20),
       "G8": (564, 540, 24), "G9": (865, 830, 35), "G10": (1100, 1000, 100)}
AGC_UNITS = ("G3", "G9", "G10")
R, TG, TT, TR, F_HP = 0.05, 0.2, 0.5, 7.0, 0.3
KP_AGC, KI_AGC, T_AGC = 2.0, 1.0, 5.0
AGC_RAMP = 0.1 / 60.0                              # p.u. of unit rating / s

UFLS = [(49.0, 0.05), (48.5, 0.05)]                # (Hz, fraction of load)
T_UFLS = 0.15                                      # relay + breaker delay

BESS = {"Bus 3": 50.0, "Bus 29": 78.0}
BESS_RAMP = 250.0                                  # MW/s

DT = 1e-3


def simulate(case="NC", t_comm=0.0, system="IEEE 39-Bus", t_end=10.0):
    """case: 'NC' (governor only), 'CC' (governor + PI-AGC),
    'PF' (governor + AGC + CNN-LSTM-dispatched BESS)."""
    H = SYSTEMS[system]
    t = np.round(np.arange(0.0, t_end + DT / 2, DT), 6)
    n = t.size
    f = np.full(n, F0)
    p_gen = {g: np.full(n, v[1], dtype=float) for g, v in GEN.items()}
    p_bess = {b: np.zeros(n) for b in BESS}
    p_shed = np.zeros(n)
    x = 0.0                                        # df / f0 (p.u.)
    st = {g: [0.0, 0.0, 0.0] for g in GEN}         # valve, HP, reheat (MW)
    agc_i = 0.0
    agc_out = {g: 0.0 for g in AGC_UNITS}
    shed_t = [None] * len(UFLS)
    shed = 0.0
    t_bess = T_DEC + t_comm
    for k in range(1, n):
        tt = t[k]
        if tt < T_EVT:
            continue
        df_hz = x * F0
        dP = 0.0
        for g, (rat, p0, head) in GEN.items():
            target = min(max(-x / R * rat, 0.0), head)
            v, hp, rh = st[g]
            v += (target - v) / TG * DT
            hp += (v - hp) / TT * DT
            rh += (hp - rh) / TR * DT
            st[g] = [v, hp, rh]
            dP += F_HP * hp + (1 - F_HP) * rh
            p_gen[g][k] = p0 + F_HP * hp + (1 - F_HP) * rh
        if case in ("CC", "PF"):
            agc_i += -df_hz * DT
            demand = KP_AGC * (-df_hz) + KI_AGC * agc_i     # p.u. of units
            for g in AGC_UNITS:
                rat, p0, head = GEN[g]
                gov = p_gen[g][k] - p0
                want = min(max(demand * rat * 0.01 * F0 / T_AGC, 0.0),
                           head - gov)
                step = np.clip(want - agc_out[g], -AGC_RAMP * rat * DT,
                               AGC_RAMP * rat * DT)
                agc_out[g] += step
                p_gen[g][k] += agc_out[g]
                dP += agc_out[g]
        if case == "PF" and tt >= t_bess:
            for b, pmax in BESS.items():
                p_bess[b][k] = min(BESS_RAMP * (tt - t_bess), pmax)
                dP += p_bess[b][k]
        for i, (fth, frac) in enumerate(UFLS):
            if shed_t[i] is None and F0 + df_hz < fth:
                shed_t[i] = tt
            if shed_t[i] is not None and tt >= shed_t[i] + T_UFLS:
                p_shed[k] += frac * S_BASE
        shed += (p_shed[k] - shed) / 0.08 * DT     # feeder disconnection
        p_shed[k] = shed
        dP += shed
        imbalance = (dP - P_TRIP) / S_BASE - D_LOAD * x
        x += imbalance / (2 * H) * DT
        f[k] = F0 * (1 + x)
    rocof = np.gradient(f, DT)
    w = int(0.1 / DT)                              # 100 ms relay window
    rocof = np.convolve(rocof, np.ones(w) / w, mode="full")[:n]
    p_gen["G6"] = np.where(t < T_EVT, 650.0, 0.0)
    return {"t": t, "f": f, "rocof": rocof, "gen": p_gen, "bess": p_bess,
            "shed": p_shed, "ufls": shed_t}


def metrics(r):
    t, f = r["t"], r["f"]
    m = t >= T_EVT
    i = np.argmin(f[m])
    above = np.where((t > t[m][i]) & (f >= 49.8))[0]
    return {"nadir": float(f[m][i]), "t_nadir": float(t[m][i]),
            "rocof_max": float(np.abs(r["rocof"][m]).max()),
            "ufls": r["ufls"], "f_end": float(f[-1]),
            "t_49_8": float(t[above[0]]) if above.size else None}


def summary(path="freq_results.json"):
    res = {"t_dec": T_DEC, "cases": {}, "systems": {}}
    for key, case, tc in [("NC", "NC", 0), ("CC", "CC", 0),
                          ("PF", "PF", 0), ("PF200", "PF", 0.2)]:
        res["cases"][key] = metrics(simulate(case, tc))
    for s in SYSTEMS:
        res["systems"][s] = {c: metrics(simulate(c, 0, s)) for c in ("NC", "PF")}
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    print(json.dumps(summary(), indent=1))
