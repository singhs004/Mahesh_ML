#!/bin/sh
# Regenerate all 11 figures and the tables document from the shared models.
set -e
cd "$(dirname "$0")"
for f in rotor_angle_timing frequency_timing delta_delta_timing \
         bus44_voltage_timing bus44_collapse_timing delay_sensitivity_timing \
         statcom_saturation_timing g10_nc_cc_pf_timing freq_deviation_timing \
         frequency_rocof_timing bess_power_timing; do
    python3 "$f.py" > /dev/null
done
python3 g10_model.py > /dev/null
python3 freq_model.py > /dev/null
python3 -c "
import json, bus44_voltage_timing as a, bus44_collapse_timing as b
def met(t, v):
    m = t >= 0.5
    return {'min': float(v[m].min()), 'below': float(((v < 0.9) & m).sum() * (t[1] - t[0])), 'end': float(v[-1])}
json.dump({'v0': 0.942,
           'wind': {'SG': met(a.t, a.v_sg), 'NC': met(a.t, a.v_nc), 'PF': met(a.t, a.cnn_lstm_svc(a.T_SVC)), 'PF200': met(a.t, a.cnn_lstm_svc(a.T_SVC + 0.2))},
           'collapse': {'NC': met(b.t, b.v_nc), 'CC': met(b.t, b.v_cc), 'PF': met(b.t, b.pf(b.T_SVC)), 'PF200': met(b.t, b.pf(b.T_SVC + 0.2))}},
          open('bus44_results.json', 'w'), indent=1)
" > /dev/null
node make_tables.js
