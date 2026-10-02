const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, ShadingType, HeadingLevel } = require('docx');

const R = JSON.parse(fs.readFileSync(__dirname + '/g10_results.json', 'utf8'));
const FR = JSON.parse(fs.readFileSync(__dirname + '/freq_results.json', 'utf8'));
const VB = JSON.parse(fs.readFileSync(__dirname + '/bus44_results.json', 'utf8'));
const hz = v => `${v.toFixed(2)} Hz`;
const pu = v => `${v.toFixed(3)} p.u.`;
const uf = c => { const st = c.ufls.map((v, i) => v === null ? null : `${i + 1}`).filter(Boolean);
  return st.length ? `stage ${st.join('+')} (t = ${c.ufls[0].toFixed(2)} s)` : 'none'; };
const row = ms => R.sweep.find(r => r.tcomm_ms === ms);
const deg = v => `${v.toFixed(1)}°`;
const pk = (ms, d) => deg(row(ms)[d].peak);
const pkd = (ms, d) => deg(row(ms)[d].peak_dev);
const FONT = 'Times New Roman';
const W = 9360; // 6.5in text width on Letter
const thin = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };

function cell(text, w, opts = {}) {
  const runs = (Array.isArray(text) ? text : [text]).map(t =>
    typeof t === 'string' ? new TextRun({ text: t, font: FONT, size: 18, bold: opts.bold })
                          : new TextRun({ font: FONT, size: 18, bold: opts.bold, ...t }));
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    shading: opts.head ? { type: ShadingType.CLEAR, fill: 'E7E6E6', color: 'auto' } : undefined,
    borders: { top: opts.head ? thin : none, bottom: opts.last || opts.head ? thin : none,
               left: none, right: none },
    children: [new Paragraph({ alignment: opts.left ? AlignmentType.LEFT : AlignmentType.CENTER,
                               children: runs })],
  });
}

function table(widths, head, rows) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({ tableHeader: true,
        children: head.map((h, i) => cell(h, widths[i], { head: true, bold: true, left: i === 0 })) }),
      ...rows.map((r, ri) => new TableRow({
        children: r.map((c, i) => cell(c, widths[i], { left: i === 0, last: ri === rows.length - 1 })) })),
    ],
  });
}

function caption(num, text) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 280, after: 0 },
      children: [new TextRun({ text: `TABLE ${num}`, font: FONT, size: 18 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
      children: [new TextRun({ text, font: FONT, size: 18, smallCaps: true })] }),
  ];
}
function note(text) {
  return new Paragraph({ spacing: { before: 60, after: 120 },
    children: [new TextRun({ text, font: FONT, size: 16, italics: true })] });
}
const sub = (base, s) => [base, { text: s, subScript: true }];

const children = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [new TextRun({ text: 'Latency Budget and Re-timed Simulation Results',
                             font: FONT, size: 28, bold: true })] }),

  ...caption('I', 'End-to-End Latency Budget of the CNN-LSTM Control Chain'),
  table([5200, 2400], ['Stage', 'Time'], [
    ['PMU observation window', '500 ms'],
    ['Preprocessing', '1.9 ms'],
    ['CNN-LSTM inference (CPU, mean)', '17.3 ms'],
    ['Decision available after onset', '519.2 ms'],
    [['Communication delay (', ...sub('T', 'comm'), ')'], '0–500 ms (examined)'],
    ['STATCOM response time', '< 30 ms'],
    ['SVC response time', '≈ 150 ms'],
    ['Total to STATCOM corrective action', ['≈ 549 ms + ', ...sub('T', 'comm')]],
    ['Total to SVC corrective action', ['≈ 669 ms + ', ...sub('T', 'comm')]],
  ]),

  ...caption('II', 'Event and Corrective-Action Instants Used in Each Figure'),
  table([2500, 1250, 1350, 1300, 1480, 1480],
    ['Figure', 'Device', 'Disturbance (s)', 'Decision (s)',
     ['Action, ', ...sub('T', 'comm'), '=0 (s)'], ['Action, ', ...sub('T', 'comm'), '=200 ms (s)']], [
    ['Rotor angle, G1/G4/G7/G10', 'STATCOM', '0.10', '0.619', '0.649', '0.849'],
    ['Rotor angle, G1/G4/G7/G10', 'SVC', '0.10', '0.619', '0.769', '—'],
    ['System frequency (G6 trip)', 'BESS', '0.50', '1.019', '1.019 + ramp', '1.219 + ramp'],
    ['Two-panel Δδ (G10)', 'STATCOM', '0.10', '0.619', '0.649', '0.849'],
    ['Two-panel Δδ (G10)', 'SVC', '0.10', '0.619', '0.769', '—'],
    ['Bus 44 voltage (wind variability)', 'SVC', '0.50', '1.019', '1.169', '1.369'],
    ['Bus 44 voltage collapse', 'SVC', '0.50', '1.019', '1.169', '1.369'],
    ['Frequency deviation (9/39/118-bus)', 'BESS', '0.50', '1.019', '1.019 + ramp', '—'],
    ['Frequency and ROCOF (UFLS)', 'BESS', '0.50', '1.019', '1.019 + ramp', '1.219 + ramp'],
    ['Generator and BESS power', 'BESS', '0.50', '1.019', '1.019 + ramp', '1.219 + ramp'],
    ['G10 NC / CC / PF', 'STATCOM', '0.10', '0.619', '0.649', '0.849'],
    ['G10 delay sensitivity', 'STATCOM', '0.10', '0.619', '0.649', ['0.649 + ', ...sub('T', 'comm')]],
    ['STATCOM saturation', 'STATCOM', '0.10', '0.619', '0.649', '—'],
  ]),
  note('Action instant = disturbance + 519.2 ms + device response (30 ms STATCOM, 150 ms SVC) + Tcomm.'),

  ...caption('III', 'Key Results of the Re-timed Figures'),
  table([2300, 3000, 2000, 2060],
    ['Figure', 'Case', 'Metric', 'Value'], [
    ['All rotor-angle figures', 'G10 pre-fault angle', 'δ0', deg(R.delta0)],
    ['', 'G1 / G4 / G7', 'Peak δ (Δδ)', `${deg(R.gens.G1.peak)} / ${deg(R.gens.G4.peak)} / ${deg(R.gens.G7.peak)} (${deg(R.gens.G1.peak_dev)} / ${deg(R.gens.G4.peak_dev)} / ${deg(R.gens.G7.peak_dev)})`],
    ['', 'G10 vs G1 / G4 / G7', 'Rise during fault (0.10–0.25 s)', `${deg(R.g10_at_clear_dev)} vs ${deg(R.gens.G1.at_clear_dev)} / ${deg(R.gens.G4.at_clear_dev)} / ${deg(R.gens.G7.at_clear_dev)}`],
    ['Rotor angle (G10)', 'No control', '180° crossing', `${R.nc_cross.toFixed(2)} s`],
    ['', ['STATCOM, ', ...sub('T', 'comm'), '=0'], 'Peak δ', pk(0, 'STATCOM')],
    ['', ['STATCOM, ', ...sub('T', 'comm'), '=200 ms'], 'Peak δ', pk(200, 'STATCOM')],
    ['', ['SVC, ', ...sub('T', 'comm'), '=0'], 'Peak δ', pk(0, 'SVC')],
    ['Two-panel Δδ (G10)', ['STATCOM, ', ...sub('T', 'comm'), '=0'], 'Peak Δδ', pkd(0, 'STATCOM')],
    ['', ['STATCOM, ', ...sub('T', 'comm'), '=200 ms'], 'Peak Δδ', pkd(200, 'STATCOM')],
    ['', ['SVC, ', ...sub('T', 'comm'), '=0'], 'Peak Δδ', pkd(0, 'SVC')],
    ['System frequency / ROCOF (G6 trip, 650 MW)', 'No control (governor)', 'Nadir / UFLS / |ROCOF|max', `${hz(FR.cases.NC.nadir)} / ${uf(FR.cases.NC)} / ${FR.cases.NC.rocof_max.toFixed(2)} Hz/s`],
    ['', 'PI-AGC (governor + AGC)', 'Nadir / UFLS / |ROCOF|max', `${hz(FR.cases.CC.nadir)} / ${uf(FR.cases.CC)} / ${FR.cases.CC.rocof_max.toFixed(2)} Hz/s`],
    ['', ['CNN-LSTM + BESS, ', ...sub('T', 'comm'), '=0'], 'Nadir / UFLS / |ROCOF|max', `${hz(FR.cases.PF.nadir)} / ${uf(FR.cases.PF)} / ${FR.cases.PF.rocof_max.toFixed(2)} Hz/s`],
    ['', ['CNN-LSTM + BESS, ', ...sub('T', 'comm'), '=200 ms'], 'Nadir / UFLS', `${hz(FR.cases.PF200.nadir)} / ${uf(FR.cases.PF200)}`],
    ['', 'No control / PI-AGC / CNN-LSTM + BESS', 'Back above 49.8 Hz', `${FR.cases.NC.t_49_8.toFixed(1)} / ${FR.cases.CC.t_49_8.toFixed(1)} / ${FR.cases.PF.t_49_8.toFixed(1)} s`],
    ['Frequency deviation (b)', '9 / 39 / 118-bus, no control', 'Δf nadir', Object.values(FR.systems).map(v => (v.NC.nadir - 50).toFixed(2)).join(' / ') + ' Hz'],
    ['', '9 / 39 / 118-bus, CNN-LSTM + BESS', 'Δf nadir', Object.values(FR.systems).map(v => (v.PF.nadir - 50).toFixed(2)).join(' / ') + ' Hz'],
    ['Generator and BESS power', 'BESS Bus 3 / Bus 29', 'Full output reached', '1.219 s / 1.331 s'],
    ['', 'BESS total', 'Delivered in first-swing window (0.50–0.70 s)', '0 MW'],
    ['Bus 44 voltage (wind variability)', 'SG-only baseline', 'Min. / time < 0.90 p.u.', `${pu(VB.wind.SG.min)} / ${VB.wind.SG.below.toFixed(1)} s`],
    ['', '50% RES, no control', 'Min. / time < 0.90 p.u.', `${pu(VB.wind.NC.min)} / ${VB.wind.NC.below.toFixed(1)} s`],
    ['', ['CNN-LSTM + SVC, ', ...sub('T', 'comm'), '=0'], 'Min. / time < 0.90 p.u.', `${pu(VB.wind.PF.min)} / ${VB.wind.PF.below.toFixed(1)} s`],
    ['', ['CNN-LSTM + SVC, ', ...sub('T', 'comm'), '=200 ms'], 'Min. / time < 0.90 p.u.', `${pu(VB.wind.PF200.min)} / ${VB.wind.PF200.below.toFixed(1)} s`],
    ['Bus 44 voltage collapse', 'All cases', 'Pre-disturbance voltage', pu(VB.v0)],
    ['', 'NC', 'Min. / at 10 s', `${pu(VB.collapse.NC.min)} / ${pu(VB.collapse.NC.end)} (collapse)`],
    ['', 'CC', 'Min. / at 10 s', `${pu(VB.collapse.CC.min)} / ${pu(VB.collapse.CC.end)}`],
    ['', ['PF, ', ...sub('T', 'comm'), '=0'], 'Min. / time < 0.90 p.u. / at 10 s', `${pu(VB.collapse.PF.min)} / ${VB.collapse.PF.below.toFixed(2)} s / ${pu(VB.collapse.PF.end)}`],
    ['', ['PF, ', ...sub('T', 'comm'), '=200 ms'], 'Min. / time < 0.90 p.u.', `${pu(VB.collapse.PF200.min)} / ${VB.collapse.PF200.below.toFixed(2)} s`],
    ['G10 NC / CC / PF', 'NC', 'Loss of synchronism', `${R.nc_cross.toFixed(2)} s`],
    ['', 'CC', 'Loss of synchronism', `${R.cc_cross.toFixed(2)} s`],
    ['', ['PF, ', ...sub('T', 'comm'), '=0'], 'Peak δ', pk(0, 'STATCOM')],
    ['', ['PF, ', ...sub('T', 'comm'), '=200 ms'], 'Peak δ', pk(200, 'STATCOM')],
    ['STATCOM saturation', `Case T alone (${R.sat.T_qmax.toFixed(0)} Mvar)`, 'Peak δ / margin', `${deg(R.sat.T_peak)} / ${(180 - R.sat.T_peak).toFixed(0)}°`],
    ['', `Combined (${R.sat.comb_demand.toFixed(0)} Mvar demand, capped 150)`, 'Peak δ / margin', `${deg(R.sat.comb_peak)} / ${(180 - R.sat.comb_peak).toFixed(0)}°`],
    ['', 'Combined', 'Saturation window', `${R.sat.window[0].toFixed(3)}–${R.sat.window[1].toFixed(2)} s`],
    ['', 'No control (combined)', '180° crossing', `${R.sat.comb_nc_cross.toFixed(2)} s`],
  ]),

  ...caption('IV', 'Communication-Delay Sensitivity of G10 Transient Stability'),
  table([1500, 2000, 2000, 2000, 1860],
    [sub('T', 'comm').length ? ['', ...sub('T', 'comm'), ' (ms)'] : '',
     'STATCOM action (s)', 'STATCOM peak δ', 'SVC action (s)', 'SVC peak δ'], [
    ...R.sweep.map(r => [String(r.tcomm_ms),
      r.STATCOM.act.toFixed(3), r.STATCOM.stable ? `${deg(r.STATCOM.peak)} (stable)` : 'Unstable',
      r.SVC.act.toFixed(3), r.SVC.stable ? `${deg(r.SVC.peak)} (stable)` : 'Unstable']),
  ]),
  note(`Critical Tcomm ≈ ${(R.crit.STATCOM * 1e3).toFixed(0)} ms for STATCOM (action at ${(R.t_act.STATCOM + R.crit.STATCOM).toFixed(3)} s) and ≈ ${(R.crit.SVC * 1e3).toFixed(0)} ms for SVC (action at ${(R.t_act.SVC + R.crit.SVC).toFixed(3)} s). All G10 values come from one shared model (g10_model.py).`),
];

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 },
    margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children }],
});
Packer.toBuffer(doc).then(b => fs.writeFileSync('/home/user/Mahesh_ML/figures/Timing_and_Results_Tables.docx', b));
