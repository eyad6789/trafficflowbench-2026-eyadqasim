"""Builds report.html (A4 pages, inline SVG charts). Render with the WebKit renderer: render.sh report.html report.pdf"""
import html, os
HERE = os.path.dirname(os.path.abspath(__file__))

def bar_chart(groups, series, colors, w=520, h=190, ymin=0.70, ymax=0.95, title='', fmt='{:.3f}', legend=True):
    """grouped vertical bars: groups = labels, series = [(name, [values])]"""
    pl, pr, pt, pb = 40, 10, 22, 34
    iw, ih = w - pl - pr, h - pt - pb
    n, m = len(groups), len(series)
    gw = iw / n; bw = gw * 0.78 / m
    y = lambda v: pt + ih * (1 - (v - ymin) / (ymax - ymin))
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica Neue, Helvetica, Arial" font-size="9">']
    if title: s.append(f'<text x="{pl}" y="12" font-size="10" font-weight="600" fill="#222">{html.escape(title)}</text>')
    t = ymin
    step = 0.05 if ymax - ymin > 0.12 else 0.02
    while t <= ymax + 1e-9:
        s.append(f'<line x1="{pl}" x2="{w - pr}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="#e3e3e3"/><text x="{pl - 4}" y="{y(t) + 3:.1f}" text-anchor="end" fill="#666">{t:.2f}</text>')
        t += step
    for gi, g in enumerate(groups):
        x0 = pl + gi * gw + gw * 0.11
        for si, (nm, vals) in enumerate(series):
            v = vals[gi]
            if v is None: continue
            x = x0 + si * bw
            s.append(f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bw - 1.5:.1f}" height="{max(0, y(ymin) - y(v)):.1f}" fill="{colors[si]}" rx="1.5"/>')
            s.append(f'<text x="{x + (bw - 1.5) / 2:.1f}" y="{y(v) - 3:.1f}" text-anchor="middle" fill="#333" font-size="7.5">{fmt.format(v)}</text>')
        s.append(f'<text x="{pl + gi * gw + gw / 2:.1f}" y="{h - pb + 12}" text-anchor="middle" fill="#333">{html.escape(g)}</text>')
    if legend:
        lx = pl
        for si, (nm, _) in enumerate(series):
            s.append(f'<rect x="{lx}" y="{h - 12}" width="8" height="8" fill="{colors[si]}" rx="1.5"/><text x="{lx + 11}" y="{h - 5}" fill="#333">{html.escape(nm)}</text>')
            lx += 14 + 5.2 * len(nm)
    s.append('</svg>')
    return ''.join(s)

def scatter(points, w=520, h=230, xr=(0.865, 0.886), yr=(0.855, 0.878)):
    pl, pr, pt, pb = 44, 12, 16, 30
    iw, ih = w - pl - pr, h - pt - pb
    X = lambda v: pl + iw * (v - xr[0]) / (xr[1] - xr[0]); Y = lambda v: pt + ih * (1 - (v - yr[0]) / (yr[1] - yr[0]))
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica Neue, Helvetica, Arial" font-size="9">']
    for t in [0.865, 0.870, 0.875, 0.880, 0.885]:
        s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{pt}" y2="{pt + ih}" stroke="#eee"/><text x="{X(t):.1f}" y="{pt + ih + 12}" text-anchor="middle" fill="#666">{t:.3f}</text>')
    for t in [0.855, 0.860, 0.865, 0.870, 0.875]:
        s.append(f'<line x1="{pl}" x2="{pl + iw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="#eee"/><text x="{pl - 4}" y="{Y(t) + 3:.1f}" text-anchor="end" fill="#666">{t:.3f}</text>')
    s.append(f'<text x="{pl + iw / 2:.1f}" y="{h - 2}" text-anchor="middle" fill="#333">public leaderboard (March)</text>')
    s.append(f'<text transform="translate(10,{pt + ih / 2:.1f}) rotate(-90)" text-anchor="middle" fill="#333">private leaderboard (April)</text>')
    for lab, xv, yv, hl, dx, dy in points:
        c = '#c0392b' if hl else '#2c5d8f'
        s.append(f'<circle cx="{X(xv):.1f}" cy="{Y(yv):.1f}" r="{4 if hl else 3}" fill="{c}"/>')
        s.append(f'<text x="{X(xv) + dx:.1f}" y="{Y(yv) + dy:.1f}" fill="{c}" font-size="8">{html.escape(lab)}</text>')
    s.append('</svg>')
    return ''.join(s)

def line_chart(xs, series, colors, w=250, h=170, ymin=0.4, ymax=1.0, title=''):
    pl, pr, pt, pb = 32, 8, 20, 30
    iw, ih = w - pl - pr, h - pt - pb
    X = lambda i: pl + iw * i / (len(xs) - 1); Y = lambda v: pt + ih * (1 - (v - ymin) / (ymax - ymin))
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica Neue, Helvetica, Arial" font-size="8.5">']
    if title: s.append(f'<text x="{pl}" y="11" font-size="9.5" font-weight="600" fill="#222">{html.escape(title)}</text>')
    t = ymin
    while t <= ymax + 1e-9:
        s.append(f'<line x1="{pl}" x2="{pl + iw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="#eee"/><text x="{pl - 3}" y="{Y(t) + 3:.1f}" text-anchor="end" fill="#666">{t:.1f}</text>')
        t += 0.1
    for i, x in enumerate(xs):
        s.append(f'<text x="{X(i):.1f}" y="{pt + ih + 11}" text-anchor="middle" fill="#333">{html.escape(x)}</text>')
    ly = h - 6
    lx = pl
    for (nm, vals), c in zip(series, colors):
        pts = ' '.join(f'{X(i):.1f},{Y(v):.1f}' for i, v in enumerate(vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="1.8"/>')
        for i, v in enumerate(vals):
            s.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="2.2" fill="{c}"/>')
        s.append(f'<rect x="{lx}" y="{ly - 7}" width="8" height="3" fill="{c}"/><text x="{lx + 11}" y="{ly - 3}" fill="#333">{html.escape(nm)}</text>')
        lx += 16 + 4.6 * len(nm)
    s.append('</svg>')
    return ''.join(s)

BLUE, RED, GREY, GREEN, GOLD = '#2c5d8f', '#c0392b', '#9aa3ad', '#2e7d5b', '#c08a2d'

fig_lb = scatter([
    ('e50 (selected)', .88236, .87520, True, -66, 3), ('em', .88408, .87372, False, 6, 10), ('full', .87677, .87254, False, 6, 3),
    ('b55', .88157, .86854, False, 6, 9), ('st', .87844, .86708, False, -16, -5), ('bm55', .87790, .86654, False, -26, 9), ('b65', .88009, .86542, False, 6, 4),
    ('one', .87190, .86388, False, 6, -3), ('og', .87206, .86362, False, 6, 8), ('on55', .87554, .86208, False, 6, 3), ('og45', .87281, .86102, False, 6, 3),
    ('ufdp50', .87156, .85621, False, 6, 3), ('uf50', .86871, .84927, False, 6, 3)], yr=(0.845, 0.878))
fig_onset = bar_chart(['k = 0', 'k = 1', 'k = 2', 'k = 3', 'k = 4', 'all'],
    [('released public model', [.744, .777, .751, .763, .739, .755]), ('A: random evidence span, no holes', [.741, .779, .780, .778, .801, .777]),
     ('final: pseudo-months, bag + CatBoost', [.775, .813, .806, .817, .808, .804])],
    [GREY, GOLD, BLUE], w=520, h=185, ymin=0.70, ymax=0.84, title='Onset IoU on March pseudo-months by window order inside the month (expected-IoU decoding)')
fig_harness = bar_chart(['fine-tuned nets', 'boosted model (final)'],
    [('calendar', [.9138, None]), ('pseudo-month', [None, .8891]), ('LB March', [.8243, .8346]), ('LB April', [.8570, .8841])],
    [GREY, GOLD, BLUE, RED], w=330, h=150, ymin=0.78, ymax=0.94, title='Ongoing IoU: local harness vs leaderboard')
fig_pers = line_chart(['T+5', 'T+10', 'T+15', 'T+20', 'T+25', 'T+30'],
    [('official', [.915, .856, .799, .765, .680, .529]), ('replica, T-1 too persistent', [.968, .95, .93, .88, .828, .71])],
    [BLUE, GOLD], w=250, h=150, ymin=0.4, ymax=1.0, title='IoU of the queue at T with the queue at T+h')

CSS = '''
@page { size: A4; margin: 0 }
* { box-sizing: border-box }
body { margin: 0; background: #fff; color: #1d1f22; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 9.6pt; line-height: 1.42 }
.page { width: 210mm; min-height: 297mm; padding: 16mm 17mm 14mm; position: relative; page-break-after: always; margin: 0 auto }
h1 { font-size: 19pt; margin: 0 0 3mm; line-height: 1.15; letter-spacing: -0.2px }
h2 { font-size: 12.5pt; margin: 6mm 0 2mm; color: #12385e; border-bottom: 1px solid #d5dde6; padding-bottom: 1mm }
h3 { font-size: 10.4pt; margin: 4mm 0 1.2mm; color: #12385e }
p { margin: 0 0 2.2mm; text-align: justify; hyphens: auto }
.sub { color: #4b5560; font-size: 10pt; margin-bottom: 4mm }
.abstract { background: #f4f7fa; border-left: 3px solid #2c5d8f; padding: 3mm 4mm; margin: 3mm 0 4mm }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 3mm; font-size: 8.7pt }
th, td { border-bottom: 1px solid #e1e5ea; padding: 1.3mm 1.6mm; text-align: left; vertical-align: top }
th { background: #eef2f6; font-weight: 600 }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums }
.fig { margin: 2mm 0 3mm; text-align: center }
.cap { font-size: 8.3pt; color: #4b5560; margin-top: 1mm; text-align: left }
.two { display: flex; gap: 6mm; align-items: flex-start }
.two > div { flex: 1 }
ul { margin: 0 0 2.2mm 4.5mm; padding: 0 }
li { margin-bottom: 1mm }
code { font-family: Menlo, monospace; font-size: 8.2pt; background: #f1f3f5; padding: 0 1mm; border-radius: 2px }
.foot { position: absolute; bottom: 7mm; left: 17mm; right: 17mm; font-size: 7.6pt; color: #8a939c; display: flex; justify-content: space-between }
.hl { color: #c0392b; font-weight: 600 }
.small { font-size: 8.5pt }
'''
def page(n, body):
    return f'<section class="page">{body}<div class="foot"><span>TrafficFlowBench 2026 — team Eyad Qasim</span><span>{n}</span></div></section>'

P = []
P.append(page(1, f'''
<h1>Learning across a change of run, inside the online rule</h1>
<div class="sub">TrafficFlowBench 2026 (IEEE Big Data Cup) — final report of team <b>Eyad Qasim</b> (Kaggle <code>eyadqasim</code>; Eyad Qasim Raheem, independent)<br>
Selected submission <code>sub_e50.csv</code>: public 0.88236, <b>private 0.87520</b> (rank 88 of 801).</div>
<div class="abstract"><p><b>Summary.</b> The two scored months are not more draws of the training months: their traffic comes from independent runs,
and the places where queues form moved. Most of our gains came from building every model so that it cannot memorise where the training run
queued, and from validating on windows that copy how the organizers select and hide the scored windows. Task 1 uses a no-memory U-Net
with time-of-day profiles taken from each scored month itself (Task 1 is offline). Task 2 uses gradient-boosted models trained on
pseudo-months that reproduce the published window selector and the 90-minute blanks after each origin, with month-to-date evidence read only
from rows stamped before the origin. On April the boosted ongoing model alone raised the ongoing IoU from 0.840 to 0.884. Every Task 2
feature is recomputed with all rows after the origin removed and checked to be identical before predictions are written.</p></div>
<h2>1. The final pipeline</h2>
<table>
<tr><th>Task</th><th>Component (package stage)</th><th>Key decisions</th></tr>
<tr><td>1 State</td><td>U-Net over day x station grids, 4 seeds (s02), mean + density step (s03)</td><td>No train-run location memory; profiles from the scored month itself; trained on train + released cells of March and April</td></tr>
<tr><td>2 Onset</td><td>LightGBM 3-seed bag + CatBoost (s04)</td><td>Pseudo-months that copy the selector and its blanks; month-to-date evidence before T; first window of a month blended with the public model; threshold 0.55, T+30 only</td></tr>
<tr><td>2 Ongoing</td><td>LightGBM, with and without evidence channels (s05)</td><td>Queue geometry and shock-speed features; no profiles, no link index, official capacity; same pseudo-months; threshold 0.5</td></tr>
<tr><td>3 Physics</td><td>scored on the Task 1 answer</td><td>Half of the log-density correction of a density head moved into flow</td></tr>
<tr><td>4 ODME</td><td>exact non-negative ridge solve (s06)</td><td>min ||Af-c||² + 0.05 ||f-b||², f ≥ 0, each split's own counts and prior</td></tr>
</table>
<h2>2. What the data taught us</h2>
<h3>2.1 The scored months come from a different run</h3>
<p>On the training months a U-Net reached an ongoing IoU of 0.92 on February windows built with the published selector, against 0.79 for the public
model, yet on the March leaderboard every model we tried landed between 0.80 and 0.83. Input statistics were indistinguishable (adversarial
validation AUC 0.47-0.54, same step noise, same 30-minute persistence), so we compared where queues form. They moved: on I5 South, links 43-45
started 38% of training onsets and none in March, while links 10-13 went from 18% to 40%; on I10 East, links 11-23 were queued 9% of the time
in training and 0.1% in March; on I405 South, links 32-39 went from 18% to 38%. The month-to-month correlation of the onset map is 0.96 inside
the training split but 0.44 between training and March (I5 South). The documentation later confirmed that validation and private were generated
independently, with their own demand draws and incidents. Anything that memorises where the training run queued — time-of-day profiles from
training months, link and corridor indices, empirical capacities — therefore misleads on the scored months.</p>
<p>The consequence is different for the two kinds of task. Task 1 is offline, so its profiles can come from the scored month itself. Task 2 is
online, so a window may only use evidence stamped before its origin: earlier days of the same month (and, for April, the training split and all of March).</p>
'''))
P.append(page(2, f'''
<h3>2.2 The scored windows follow a published selector, and the data around them is blanked</h3>
<ul>
<li><b>Onset windows are first feasible origins.</b> In all 40 official June training windows the first queued cell appears at exactly T+30
(one has none, by observation noise), so only the T+30 step can be queued.</li>
<li><b>Windows are the earliest qualifying origins of the month, at least 6 hours apart across both conditions.</b> Each corridor gets roughly
one onset (around 09:00) and one ongoing window (around 15:00) per day, so all scored windows sit in the first 1-16 days of the month: 21% of
onset windows have less than one day of month-to-date evidence, 22% less than two.</li>
<li><b>The 90 minutes after every origin are blank.</b> Rows T+1 to T+18 of every scored window are missing in the released layer (0% coverage),
while the 60-minute histories are released at 83% coverage against 58% for the ordinary masked layer. A later window therefore never sees how the
earlier queues of the month began.</li>
</ul>
<h3>2.3 Local validation must copy 2.2, or it lies</h3>
<p>Our first harness used calendar windows of March, read through the masked layer and outside the scored windows. It said that fine-tuning
networks on earlier days of the same month lifted the ongoing IoU from 0.873 to 0.914; the leaderboard said 0.824. It also preferred
expected-IoU decoding for onset by 0.05; the March leaderboard preferred a 0.55 threshold. Calendar windows are mostly mid-event, with clean
evidence about the same queue, and their 58% coverage rewards models that impute well. Scored windows sit at fixed places in the month, follow
blanks and carry 83% coverage. We rebuilt validation as <b>pseudo-months</b>: from a start day, run the selector forward, take its first five
windows of each condition, blank rows T+1 to T+18 after each, and read evidence only from rows before each origin: with exact labels on
February (fitted on June-December) and across the run change on March (fitted on train).</p>
<div class="two"><div class="fig">{fig_harness}<div class="cap">Figure 1. The calendar harness overstated the fine-tuned networks by 0.09; the pseudo-month harness ranked the
boosted model correctly. On April the boosted model is the best part of our final file.</div></div>
<div class="fig">{fig_pers}<div class="cap">Figure 2. How long the queue at T persists in official ongoing windows (June, train labels), and in replica windows chosen
later than the 6-hour boundary because T-1 failed only on the persistence limit.</div></div></div>
<h2>3. Task 1 (state) and Task 3 (physics)</h2>
<p><b>Model.</b> A 2-D U-Net (48-96-192-192 channels, GroupNorm, GELU, three poolings) over one-day grids of 288 steps x stations. Inputs are the
released anchors as deviations from a time-of-day profile (speed, flow per lane, occupancy density per lane), ineligible-with-value channels, the
profile itself, ramps, static link data, time of day, mask rate and anchor/target flags. Training uses the official train masks plus random extra
anchor dropout (0-35%), so dropped anchors become extra targets; the loss is MSE on speed, MSE on flow/512, and 4 x L1 on the accumulation N = kL.</p>
<p><b>Profiles from the month itself.</b> The profile of every month — train, March, April — is a leave-one-day-out weekday/weekend profile of that
month's released cells, shrunk toward the all-days profile (κ = 6) and then the link mean. With training-run profiles the March held-out score was
0.9368 (speed RMSE 1.84 km/h); month-own profiles without retraining gave 0.9420; retraining with month-own profiles on train plus the released cells
of March and April (20% of them held out, counted twice) gave <b>0.9487</b> on March and 0.9482 on April (speed RMSE 1.33, flow 29.6 vph/lane), the
level reached on February before the run change. Four seeds are averaged. On the leaderboard this state alone added +0.0056 public and +0.0061 private.</p>
<p><b>Physics.</b> A density head predicts log k; moving half of its gap to log(q/v) into the flow lowered the conservation proxy from 0.236
to 0.229 on held-out cells for a 1.4% rise in flow RMSE. Exact answers stop near 0.96 on S_physics because the flux is projected onto noisy observations.</p>
'''))
P.append(page(3, f'''
<h2>4. Task 2 — onset</h2>
<p><b>Training data.</b> 2,373 onset events of the training split (first origins of each feasible run, labels from the unmasked layer at T+30), placed
in 5,274 pseudo-month rows: for each start day (every second day) the selector picks the first five onset and ongoing windows, rows T+1..T+18 after
each are blanked, histories are shown, and the evidence for window k is read from the start day up to its origin. The window order k (0-4) and the
evidence span therefore match the scored months.</p>
<p><b>Features.</b> The 83 features of the public onset model (recent speeds and flows, lags, neighbours, milepost neighbours) plus 46 evidence
channels: queue rate per link over the evidence span and near this time of day on earlier days, today's rate so far, share of earlier onset events
that hit the link — each earlier event read over its first six rows, since one row has only 58% coverage — the link's usual first-queue time,
the share of time the link is the head of a queue (a queue grows upstream, so the head stays at the bottleneck even when the start was blanked),
flows and speeds near this time on earlier days, and the same channels one to three links downstream. Two copies are trained, with and without
the panel number, link index and milepost.</p>
<p><b>Models and decoding.</b> LightGBM (learning rate 0.04, 15 leaves, 1,500 rounds) bagged over three seeds, and CatBoost (depth 7, 1,500
iterations), averaged with equal weight over both copies. For the first window of a corridor in a month there is no evidence yet, so its probability
is blended 0.6 / 0.4 with the released public model. A link is predicted queued at T+30 when p ≥ 0.55; if none is, the most likely link is.</p>
<div class="fig">{fig_onset}<div class="cap">Figure 3. March pseudo-months, fitted on train only. Training on pseudo-months with blanks (rather than random evidence spans
without blanks) is worth about 0.02 of IoU, most of it after the first window of the month.</div></div>
<table>
<tr><th>Onset model on March pseudo-months (expected-IoU decoding)</th><th class="n">IoU</th></tr>
<tr><td>Released public model</td><td class="n">0.755</td></tr>
<tr><td>A: evidence over a random span of 0-14 days, no blanks (the first version we submitted)</td><td class="n">0.777</td></tr>
<tr><td>B: trained on pseudo-months with blanks, same features</td><td class="n">0.783</td></tr>
<tr><td>C: + wide time-of-day rate, days with a queue, first-queue time, queue heads, neighbours</td><td class="n">0.794</td></tr>
<tr><td>D: + events read over six rows, last event, time-of-day kernel, flows and speeds near this time</td><td class="n">0.799</td></tr>
<tr><td>Final: LightGBM bag + CatBoost, first window blended with the public model</td><td class="n">0.804</td></tr>
</table>
<p><b>Error analysis.</b> Of 759 March pseudo-windows, 564 are exact, 147 partial and 8.3% score zero. In the zero windows the model's best
guess is usually 26 to 133 links away from the true queue and its probability on the true link averages 0.09: the queue starts at a bottleneck the
month has not shown before, which is what an incident looks like (validation carries five, private seven). Partial windows are mostly a set of one or
two links predicted as two or three. On I5 South the onset map changed the most and the IoU is lowest (0.55).</p>
<p><b>Decoding moved between months.</b> On March, 0.55 beat expected-IoU by 0.024 of onset IoU; on April expected-IoU would have been 0.012
better. We chose 0.55 on March, the only month the rules allow for selecting April's method.</p>
'''))
P.append(page(4, f'''
<h2>5. Task 2 — ongoing</h2>
<p><b>Why not the networks.</b> A 1-D U-Net over the corridor with a space-time stage reached 0.92 on February selector windows, and online
fine-tuning on earlier windows of the same month looked even better on the calendar harness, but on the leaderboard the networks agreed with the
public gradient-boosted models on 96% of the scored cells and scored 0.80-0.83 like them. A network sees the static channels of a corridor and can
recognise where it is; that knowledge belongs to the training run.</p>
<p><b>Model.</b> LightGBM (learning rate 0.03, 127 leaves, minimum 100 rows per leaf, feature and bagging fractions 0.7, 1,400 rounds), one row per
link and horizon step. Features come from the gap-filled 60-minute history and the released row at T: speed ratios at several lags, trends and a
30-minute linear projection; flow and occupancy; neighbours one to six links away; the distance to the nearest queued link downstream and upstream
now, 15 and 30 minutes ago, and their changes; the tail speed of the queue from the shock-wave relation w = (q_up - q_in)/(k_up - k_in), the time
for the tail to reach each link and for a queue segment to clear, projected to every horizon step; queued-block length and position; ramps. There
are no profiles, no link or corridor index, and capacity is the official value. Training rows: 8.3 million (link, step) cells of official-like
windows from pseudo-months started on every day of the training split. A second model adds the 46 evidence channels of the onset model; the
submission averages both and predicts queued where p ≥ 0.5 (a window with no queued cell gets its expected-IoU set).</p>
<table>
<tr><th>Ongoing IoU</th><th class="n">Feb exact labels</th><th class="n">March pseudo-months</th><th class="n">public LB</th><th class="n">private LB</th></tr>
<tr><td>Public ongoing model (anubisyy)</td><td class="n">0.823</td><td class="n">0.806</td><td class="n">≈0.80</td><td class="n">—</td></tr>
<tr><td>Reference blend we used until Nov 5 (public + our first GBM, 0.45)</td><td class="n">—</td><td class="n">—</td><td class="n">0.829</td><td class="n">0.840</td></tr>
<tr><td>Fine-tuned no-memory U-Nets + 0.15 public</td><td class="n">—</td><td class="n">0.914 (calendar)</td><td class="n">0.824</td><td class="n">0.857</td></tr>
<tr><td><b>Boosted model, mean of the two versions (final)</b></td><td class="n">0.891</td><td class="n">0.889</td><td class="n"><b>0.835</b></td><td class="n"><b>0.884</b></td></tr>
</table>
<p class="small">Leaderboard IoUs are derived from paired submissions that differ only in the ongoing rows (Δ score / 0.15) and an all-zero ongoing probe (0.8293 for the reference on March).</p>
<p><b>What the selector reveals.</b> Because the selector keeps the earliest qualifying origin at least six hours after the previous window, an
origin that comes later than that boundary tells something about the horizon: the previous origin did not qualify, often because the queue would
have persisted too well. In replica windows of this kind the queue at T matches the queue at T+5 with IoU 0.97 and at T+30 with 0.71 (Figure 2).
We added the gap since the previous window to the features, which the rules allow, but our replica of the selector reproduces only some of the official
June ongoing origins (onset origins are reproduced 4-5 out of 5 per corridor), and the feature did not help on February. We left it out.</p>
<h2>6. Task 4 — ODME</h2>
<p>For each corridor and split we solve min ||Af - c||² + 0.05 ||f - b||² over f ≥ 0 with the split's own released counts c and weak prior b,
measured links only, by exact non-negative least squares. It reproduces the public 0.86517 pipeline's L-BFGS-B answer to a mean relative difference of
1.4e-6. Two leaderboard probes that moved the flows by a quarter (a prior-weighted solve and a multiplicative KL fit) lost 0.13 and 0.16 of S_ODME,
about the distance moved, which only happens when the solution already sits on the reference.</p>
'''))
P.append(page(5, f'''
<h2>7. Results</h2>
<div class="fig">{fig_lb}<div class="cap">Figure 4. Our submissions of the last four days, public against private. The public leaderboard (40 windows per condition) ranked
our two best files the wrong way round; the boosted ongoing model is the difference between the top two.</div></div>

<table>
<tr><th>Submission</th><th>What changed</th><th class="n">public</th><th class="n">private</th></tr>
<tr><td>og45 (Nov 3)</td><td>previous best: U-Net + GBM state with training-run profiles, public onset, blended ongoing</td><td class="n">0.87281</td><td class="n">0.86102</td></tr>
<tr><td>st (Nov 5)</td><td>+ no-memory U-Net state with month-own profiles (2 seeds)</td><td class="n">0.87844</td><td class="n">0.86708</td></tr>
<tr><td>b55 (Nov 6)</td><td>+ 4 seeds, + onset bag + CatBoost at 0.55</td><td class="n">0.88157</td><td class="n">0.86854</td></tr>
<tr><td><b>e50 (Nov 6, selected)</b></td><td><b>+ boosted ongoing model</b></td><td class="n">0.88236</td><td class="n"><b>0.87520</b></td></tr>
<tr><td>em (Nov 6)</td><td>ongoing half boosted model, half reference blend</td><td class="n"><b>0.88408</b></td><td class="n">0.87372</td></tr>
</table>
<h2>8. What did not work</h2>
<table>
<tr><th>Idea</th><th>Evidence</th></tr>
<tr><td>Training-run time-of-day profiles, link indices, empirical capacity</td><td>Help on February, hurt on March (2.1)</td></tr>
<tr><td>Online fine-tuning of networks on the same month</td><td>+0.04 on the calendar harness, −0.005 on the March LB (+0.017 on April)</td></tr>
<tr><td>Previous-month evidence for the first window</td><td>No better than same-day evidence across a run change (0.762-0.765 vs 0.766-0.768)</td></tr>
<tr><td>Refitting onset models on the same month's earlier events</td><td>+0.004-0.016, mostly late in the month where no scored window sits</td></tr>
<tr><td>Growing onset sets upstream; lower onset thresholds</td><td>−0.007 to −0.011 on the March LB</td></tr>
<tr><td>Spatial or temporal smoothing of ongoing probabilities</td><td>−0.06 (space) / ±0.002 (time) on March pseudo-months</td></tr>
<tr><td>Selector gap features for ongoing</td><td>0.886 vs 0.887 on February (replica of the selector too rough)</td></tr>
<tr><td>Exact ODME solve</td><td>identical to the public solver to 1.4e-6</td></tr>
</table>
'''))
P.append(page(6, '''
<h2>9. The online rule</h2>
<p>For Task 2 every input is stamped at or before the origin: the 60-minute history, the released row at T, and evidence channels computed
from prefix sums that read only rows before T (earlier days of the same split; the origins of earlier windows from window_index). All models are
fitted on the training split only; thresholds and blend weights were chosen on the training split and the March leaderboard. Stage 04 and stage 05
of the package repeat the organizers' check themselves: for each of the 80 scored windows of each condition (40 in March, 40 in April) they set every row after the origin to
missing in all arrays (speed, flow, occupancy, ramps, eligibility), recompute the window's features and stop if any value differs by more than
floating-point rounding (relative tolerance 1e-5). All windows pass, so the predictions cannot change when those rows are removed.</p>
<p><b>Disclosure.</b> On November 3 we scored the public model and one U-Net on April calendar windows once, as a diagnostic of the month shift
(it showed the same drop as March). No component, threshold or weight of the final file was chosen from it; the decisions it could have touched
(removing location memory, month-own Task 1 profiles) were already supported by the training split and March. Task 1 and Task 4 use whole months,
which the rules allow for offline tasks.</p>
<h2>10. Reproducibility</h2>
<p><code>python run_all.py --data &lt;competition input&gt; --assets &lt;public assets&gt; --out run</code> runs seven stages: s01 data preparation,
s06 ODME, s04 onset, s05 ongoing, s02 four U-Net seeds (2026, 7, 11, 23), s03 state merge and s07 submission, which writes
<code>run/s07_submission/submission.csv</code> and, given a copy of our submitted file, compares the two cell by cell. Runtime on a Kaggle machine with
4 CPUs, one T4 GPU and 30 GB RAM: about 7 hours (4.5 for the U-Nets). The stages are plain scripts and also run unchanged as Kaggle notebooks.
The Task 2 stages read each split's dates from the data, and <code>run_all.py --task2-only</code> runs only them (stages 01, 04, 05, 07), for a newly
generated month in the same layout.</p>
<p><b>External data and code.</b> No external data. The public 0.86517 pipeline of <i>anubisyy</i> (dataset trafficflowbench-086517-public-assets,
Apache-2.0): its onset and ongoing checkpoints (onset probabilities for the first window of a month; the ongoing model as comparison only) and its
feature code as written out by its companion notebook (config.py, queue_features.py, queue_ongoing.py, base_panel.py, unchanged in third_party/). Libraries: numpy, pandas, scipy, pyarrow, LightGBM, CatBoost,
PyTorch. Our code is released under Apache-2.0.</p>
'''))
HTML = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>TrafficFlowBench 2026 report</title><style>{CSS}</style></head><body>{"".join(P)}</body></html>'
open(os.path.join(HERE, 'report.html'), 'w').write(HTML)
print('pages', len(P))
