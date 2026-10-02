const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, ShadingType, HeadingLevel } = require('docx');

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
    ['System frequency', 'SVC', '0.50', '1.019', '1.169', '1.369'],
    ['Two-panel Δδ (G10)', 'STATCOM', '0.10', '0.619', '0.649', '0.849'],
    ['Two-panel Δδ (G10)', 'SVC', '0.10', '0.619', '0.769', '—'],
    ['Bus 44 voltage (wind variability)', 'SVC', '0.50', '1.019', '1.169', '1.369'],
    ['Bus 44 voltage collapse', 'SVC', '0.50', '1.019', '1.169', '1.369'],
    ['Frequency deviation (9/39/118-bus)', 'SVC', '0.50', '1.019', '1.169', '—'],
    ['Frequency and ROCOF (UFLS)', 'SVC', '0.50', '1.019', '1.169', '1.369'],
    ['Generator and BESS power', 'BESS', '0.50', '1.019', '1.019 (ramp 250 MW/s)', '1.219'],
    ['G10 NC / CC / PF', 'STATCOM', '0.10', '0.619', '0.649', '0.849'],
    ['G10 delay sensitivity', 'STATCOM', '0.10', '0.619', '0.649', ['0.649 + ', ...sub('T', 'comm')]],
    ['STATCOM saturation', 'STATCOM', '0.25', '0.769', '0.799', '—'],
  ]),
  note('Action instant = disturbance + 519.2 ms + device response (30 ms STATCOM, 150 ms SVC) + Tcomm.'),

  ...caption('III', 'Key Results of the Re-timed Figures'),
  table([2300, 3000, 2000, 2060],
    ['Figure', 'Case', 'Metric', 'Value'], [
    ['Rotor angle (G10)', 'No control', 'Peak δ', 'Loses synchronism'],
    ['', ['STATCOM, ', ...sub('T', 'comm'), '=0'], 'Peak δ', '52.9°'],
    ['', ['STATCOM, ', ...sub('T', 'comm'), '=200 ms'], 'Peak δ', '117.4°'],
    ['', ['SVC, ', ...sub('T', 'comm'), '=0'], 'Peak δ', '97.1°'],
    ['System frequency', 'No control', 'Nadir', '48.30 Hz'],
    ['', 'Conventional PI-AGC', 'Nadir', '49.10 Hz'],
    ['', ['CNN-LSTM + SVC, ', ...sub('T', 'comm'), '=0 / 200 ms'], 'Nadir', '48.30 Hz (before action)'],
    ['Two-panel Δδ (G10)', ['STATCOM, ', ...sub('T', 'comm'), '=0'], 'Peak Δδ', '20.7°'],
    ['', ['STATCOM, ', ...sub('T', 'comm'), '=200 ms'], 'Peak Δδ', '43.1°'],
    ['', ['SVC, ', ...sub('T', 'comm'), '=0'], 'Peak Δδ', '34.6°'],
    ['Bus 44 voltage (wind variability)', 'SG-only baseline', 'Nadir / time < 0.90 p.u.', '0.889 p.u. / 1.0 s'],
    ['', '50% RES, no control', 'Nadir / time < 0.90 p.u.', '0.831 p.u. / 4.2 s'],
    ['', ['CNN-LSTM + SVC, ', ...sub('T', 'comm'), '=0'], 'Nadir / time < 0.90 p.u.', '0.859 p.u. / 1.4 s'],
    ['', ['CNN-LSTM + SVC, ', ...sub('T', 'comm'), '=200 ms'], 'Nadir / time < 0.90 p.u.', '0.843 p.u. / 1.4 s'],
    ['Bus 44 voltage collapse', 'NC', 'Final state', 'Collapse < 0.60 p.u.'],
    ['', 'CC', 'Nadir / final', '0.785 / 0.86 p.u.'],
    ['', ['PF, ', ...sub('T', 'comm'), '=0'], 'Nadir / time < 0.90 p.u.', '0.858 p.u. / 1.0 s'],
    ['', ['PF, ', ...sub('T', 'comm'), '=200 ms'], 'Nadir / time < 0.90 p.u.', '0.827 p.u. / 1.5 s'],
    ['Frequency deviation (a), 39-bus', 'No control / PI-AGC', 'Δf nadir', '−1.70 / −0.90 Hz'],
    ['', 'CNN-LSTM + SVC', 'Δf nadir', '−1.70 Hz (before action)'],
    ['Frequency deviation (b)', '9 / 39 / 118-bus, CNN-LSTM + SVC', 'Δf nadir', '−0.57 / −0.67 / −0.75 Hz'],
    ['', '9 / 39 / 118-bus, second dip', 'No control → CNN-LSTM', '−0.27/−0.31/−0.35 → ≈ −0.04 Hz'],
    ['Frequency and ROCOF (UFLS)', 'NC', 'Nadir / UFLS / |ROCOF|max', '48.02 Hz / stages 1+2 / 2.48 Hz/s'],
    ['', 'CC/PI-AGC', 'Nadir / UFLS / |ROCOF|max', '48.22 Hz / stages 1+2 / 2.45 Hz/s'],
    ['', ['PF, ', ...sub('T', 'comm'), '=0'], 'Nadir / UFLS / |ROCOF|max', '48.28 Hz / stages 1+2 / 2.48 Hz/s'],
    ['', ['PF, ', ...sub('T', 'comm'), '=200 ms'], 'Nadir / UFLS / |ROCOF|max', '48.09 Hz / stages 1+2 / 2.48 Hz/s'],
    ['', 'All cases', 'UFLS-1 / UFLS-2 pickup (NC)', '0.92 s / 1.13 s (before SVC action)'],
    ['Generator and BESS power', 'BESS Bus 3 / Bus 29', 'Full output reached', '1.219 s / 1.331 s'],
    ['', 'BESS total', 'Delivered in first-swing window (0.50–0.70 s)', '0 MW'],
    ['G10 NC / CC / PF', 'NC', 'Loss of synchronism', '1.44 s'],
    ['', 'CC', 'Loss of synchronism', '1.68 s'],
    ['', ['PF, ', ...sub('T', 'comm'), '=0'], 'Peak δ', '51.6°'],
    ['', ['PF, ', ...sub('T', 'comm'), '=200 ms'], 'Peak δ', '79.5°'],
    ['STATCOM saturation', 'Case T alone (140 Mvar)', 'Peak δ / margin', '64.9° / 115°'],
    ['', 'Combined (205 Mvar demand, capped 150)', 'Peak δ / margin', '81.2° / 99°'],
    ['', 'Combined', 'Saturation window', '0.809–1.59 s'],
    ['', 'No control', '180° crossing', '1.70 s'],
  ]),

  ...caption('IV', 'Communication-Delay Sensitivity of G10 Transient Stability'),
  table([1500, 2000, 2000, 2000, 1860],
    [sub('T', 'comm').length ? ['', ...sub('T', 'comm'), ' (ms)'] : '',
     'STATCOM action (s)', 'STATCOM peak δ', 'SVC action (s)', 'SVC peak δ'], [
    ['0', '0.649', '52.9° (stable)', '0.769', '97.1° (stable)'],
    ['50', '0.699', '66.2° (stable)', '0.819', '116.4° (stable)'],
    ['100', '0.749', '81.8° (stable)', '0.869', '136.2° (stable)'],
    ['150', '0.799', '99.3° (stable)', '0.919', '156.1° (stable)'],
    ['200', '0.849', '117.4° (stable)', '0.969', 'Unstable'],
    ['250', '0.899', '135.3° (stable)', '1.019', 'Unstable'],
    ['300', '0.949', '152.0° (stable)', '1.069', 'Unstable'],
    ['350', '0.999', 'Unstable', '1.119', 'Unstable'],
    ['400–500', '1.049–1.149', 'Unstable', '1.169–1.269', 'Unstable'],
  ]),
  note('Critical Tcomm ≈ 342 ms for STATCOM (≈ 891 ms total after fault onset) and ≈ 150–200 ms for SVC.'),
];

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 },
    margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children }],
});
Packer.toBuffer(doc).then(b => fs.writeFileSync('/home/user/Mahesh_ML/figures/Timing_and_Results_Tables.docx', b));
