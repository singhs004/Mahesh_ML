# Figure data (model output, not Simulink)

These files hold the exact time series plotted in the figures. They come
from the simplified Python models in `figures/`, **not** from Simulink or
any detailed power-system simulation. Each `.csv` starts with `#` comment
lines describing the case; each `.mat` holds the same columns as variables
plus a `note` string.

| File | Contents |
|---|---|
| `caseT_rotor_angles` | time, G1/G4/G7, G10 for NC, CC, PF-STATCOM (T_comm 0 and 200 ms), PF-SVC; Bus 16 STATCOM Q |
| `frequency_case` | time, frequency and ROCOF for NC, PI-AGC, CNN-LSTM+BESS (T_comm 0 and 200 ms); BESS Bus 3/29 power; G1/G3/G9/G10/G6 power; UFLS shed |
| `bus44_voltage_collapse_case` | time, Bus 44 voltage for NC, CC, PF (T_comm 0 and 200 ms) |
| `bus44_voltage_wind_case` | time, Bus 44 voltage for SG-only, 50% RES NC, PF (T_comm 0 and 200 ms) |

Regenerate with `python3 export_data.py` from `figures/`.
