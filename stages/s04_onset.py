"""Stage 04 - Task 2 onset. LightGBM (3-seed bag) and CatBoost on the 83 public onset features plus month-to-date evidence read only from
rows stamped before T, trained on train-split pseudo-months that copy the official selector (first feasible origins, 6 h apart, the 90
minutes after each origin hidden). Also prints the March pseudo-month check and writes onset_probs.csv (per scored link: model
probabilities, the released public onset model's probability, window order k). Online-rule check: every scored window's features are
recomputed with all rows after its origin removed and must be identical. Public feature code and onset checkpoints: anubisyy (Apache-2.0)."""
# ---- paths: Kaggle by default; run_all.py points these at local folders -------------------------------------
import os as _os, glob as _glob
TFB_ROOTS = [r for r in _os.environ.get("TFB_INPUT", "/kaggle/input").split(_os.pathsep) if r]
TFB_OUT = _os.environ.get("TFB_OUT", "/kaggle/working"); _os.makedirs(TFB_OUT, exist_ok=True)
def G(pat):
    """every file matching pat under the input roots (recursive), sorted"""
    return sorted(set(p for r in TFB_ROOTS for p in _glob.glob(_os.path.join(r, "**", pat), recursive=True)))
# ----------------------------------------------------------------------------------------------------------------
import glob, json, os, shutil, sys, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings('ignore')
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
def one(pat):
    f = G(pat); assert len(f) == 1, (pat, f); return f[0]
DATA = os.path.dirname(one('kaggle_public/submission_key.csv')); ASSETS = os.path.dirname(one('ongoing_lgb.txt')); PREP = os.path.dirname(one('prep/meta.json'))
SRC = _os.path.join(TFB_OUT, 'src'); os.makedirs(SRC, exist_ok=True)
for f in ('config.py', 'queue_features.py'):
    shutil.copy([c for c in G(f) if 'public_pipeline' in c or 'catboost' in c][0], f'{SRC}/{f}')     # public pipeline feature code (anubisyy, Apache-2.0)
os.environ.update(TFB_DATA_ROOT=DATA, TFB_OUTPUT_ROOT=_os.path.join(TFB_OUT, 'tmp'), TFB_THREADS='4'); sys.path.insert(0, SRC)
from catboost import CatBoostClassifier
from queue_features import Panel as Pub, panel_names, HISTORY, HORIZON
META = json.load(open(f'{PREP}/meta.json'))
pub_lgb = lgb.Booster(model_file=f'{ASSETS}/onset_lgb.txt'); pub_cat = CatBoostClassifier(); pub_cat.load_model(f'{ASSETS}/onset.cbm')
RNG = np.random.default_rng(5)
START = {s: pd.Timestamp(v[0], tz='UTC') for s, v in META['splits'].items()}; NTRAIN = int(META['splits']['train'][1])     # first day of each split and train days, read from the data by stage 01
def cum(a): return np.concatenate([np.zeros((1, a.shape[1]), np.int32), np.cumsum(a, 0, dtype=np.int32)])

class View:
    """masked-layer view of a span: prefix sums for fast statistics of rows stamped < T"""
    def __init__(s, V, Q, E, cut, cap, dn, up):
        kn = np.isfinite(V) & E; q = (np.nan_to_num(V, nan=1e9) <= cut) & kn
        s.kn, s.q, s.cap, s.n, s.dn, s.up = kn, q, cap, len(V), dn, up
        s.dn2 = np.where(dn >= 0, dn[np.maximum(dn, 0)], -1); s.up2 = np.where(up >= 0, up[np.maximum(up, 0)], -1)
        s.CQ = cum(q); s.CK = cum(kn)
        L = q.shape[1]; nd = len(V) // 288
        Qk = np.where(kn, Q, np.nan)
        s.DM = np.nan_to_num(np.nanmax(Qk.reshape(nd, 288, -1), axis=1), nan=0.0)     # per-day max flow per link
        s.Qk = Qk
        a = q.any(1); ev = []; run = 0
        for t in range(len(a)):
            if a[t]:
                if run >= 18: ev.append(t)
                run = 0
            else:
                run += 1
        s.ev = np.array(ev, np.int64); s.evq = q[s.ev] if len(ev) else np.zeros((0, L), bool)
        # a queue head: queued link whose next link downstream is known and free. Queues grow upstream, so the head stays at the bottleneck even when the start of the queue was hidden
        dd = np.maximum(dn, 0); h = q & kn[:, dd] & ~q[:, dd] & (dn >= 0)[None, :]
        s.CH = cum(h); s.evh = h[s.ev] if len(ev) else np.zeros((0, L), bool)
        qd = q.reshape(nd, 288, L); s.DQ = qd.any(1); fq = qd.argmax(1).astype(np.float32); fq[~s.DQ] = np.nan; s.FQ = fq
        s.dn3 = np.where(s.dn2 >= 0, dn[np.maximum(s.dn2, 0)], -1); s.cut = cut
        s.CF = np.concatenate([np.zeros((1, L)), np.cumsum(np.where(kn, np.nan_to_num(Q), 0.0), 0)]); s.CV = np.concatenate([np.zeros((1, L)), np.cumsum(np.where(kn, np.nan_to_num(V), 0.0), 0)])
        if len(ev):
            s.evu = np.stack([q[t:t + 6].any(0) for t in s.ev]); s.evhu = np.stack([h[t:t + 6].any(0) for t in s.ev])
        else:
            s.evu = np.zeros((0, L), bool); s.evhu = np.zeros((0, L), bool)
    def nb(s, v, i): return np.where(i >= 0, v[np.maximum(i, 0)], -1.0)
    def feats(s, T, ev_start):
        L = s.q.shape[1]; d0 = T // 288 * 288; ds = ev_start // 288; d = T // 288
        c1 = (s.CQ[T] - s.CQ[ev_start]) / np.maximum(s.CK[T] - s.CK[ev_start], 1)
        num = np.zeros(L); den = np.zeros(L); t30 = T + 6; numw = np.zeros(L); denw = np.zeros(L)
        for j in range(1, 15):
            c = t30 - 288 * j
            if c + 12 <= ev_start: break
            lo = max(c - 18, ev_start); numw += s.CQ[c + 12] - s.CQ[lo]; denw += s.CK[c + 12] - s.CK[lo]
            if c - 6 < ev_start: continue
            num += s.CQ[c + 7] - s.CQ[c - 6]; den += s.CK[c + 7] - s.CK[c - 6]
        c3 = np.where(den > 0, num / np.maximum(den, 1), -1.0)
        c4 = (s.CQ[T] - s.CQ[max(d0, ev_start)]) / np.maximum(s.CK[T] - s.CK[max(d0, ev_start)], 1) if T > max(d0, ev_start) else np.zeros(L)
        days = (T - ev_start) / 288.0
        i = np.searchsorted(s.ev, ev_start); k = np.searchsorted(s.ev, T)      # earlier onset events with first queue before T
        nev = k - i
        c5 = s.evq[i:k].mean(0) if nev > 0 else np.full(L, -1.0)
        dm = s.DM[ds:d].max(0) if d > ds else np.zeros(L)
        today = s.Qk[d0:T + 1]
        tm = np.nanmax(today, 0) if len(today) else np.zeros(L)
        mx = np.maximum(dm, np.nan_to_num(tm, nan=0.0)) / s.cap
        m = np.column_stack([c1, c3, c4, np.full(L, days), c5, np.full(L, nev), mx]).astype(np.float32)
        c6 = np.where(denw > 0, numw / np.maximum(denw, 1), -1.0)
        if d > ds:
            c7 = s.DQ[ds:d].mean(0); fq = s.FQ[ds:d]; tod = (T + 6) % 288
            c8 = tod - np.nanmedian(fq, 0); c8b = tod - np.nanmin(fq, 0)
        else:
            c7 = np.full(L, -1.0); c8 = np.full(L, np.nan); c8b = np.full(L, np.nan)
        c9 = (s.CH[T] - s.CH[ev_start]) / np.maximum(s.CK[T] - s.CK[ev_start], 1)
        c10 = s.evh[i:k].mean(0) if nev > 0 else np.full(L, -1.0)
        nb = s.nb
        x = np.column_stack([c6, c7, c8, c8b, c9, c10, nb(c1, s.dn), nb(c1, s.up), nb(c1, s.dn2), nb(c1, s.up2), nb(c9, s.dn), nb(c9, s.up), nb(c9, s.dn2), nb(c9, s.up2),
                             nb(c3, s.dn), nb(c3, s.up), nb(c5, s.dn), nb(c5, s.up), nb(c10, s.dn), nb(c10, s.up), nb(c6, s.dn), nb(c6, s.up)]).astype(np.float32)
        k2 = np.searchsorted(s.ev + 6, T, side='right'); ne = max(k2 - i, 0)
        if ne > 0:
            eu = s.evu[i:k2]; ehu = s.evhu[i:k2]; u1 = eu.mean(0); u2 = ehu.mean(0); u3 = eu[-1].astype(np.float32)
            dt = ((s.ev[i:k2] % 288) - ((T + 6) % 288)).astype(np.float64); wt = np.exp(-.5 * (dt / 24.0) ** 2)
            u4 = (wt[:, None] * eu).sum(0) / max(wt.sum(), 1e-6); u5 = np.full(L, wt.sum())
        else:
            u1 = u2 = u3 = u4 = np.full(L, -1.0); u5 = np.zeros(L)
        if d > ds:
            dq = s.DQ[ds:d]; has = dq.any(0); last = dq.shape[0] - 1 - np.argmax(dq[::-1], 0); u6 = np.where(has, dq.shape[0] - last, -1.0)
        else:
            u6 = np.full(L, -1.0)
        fn = np.zeros(L); vn = np.zeros(L); kd = np.zeros(L); vm = []
        for j in range(1, 15):
            c = t30 - 288 * j
            if c - 6 < ev_start: break
            df = s.CF[c + 7] - s.CF[c - 6]; dv = s.CV[c + 7] - s.CV[c - 6]; dk = s.CK[c + 7] - s.CK[c - 6]
            fn += df; vn += dv; kd += dk; vm.append(np.where(dk > 0, dv / np.maximum(dk, 1), np.nan))
        f1 = np.where(kd > 0, fn / np.maximum(kd, 1) / s.cap, -1.0); v1 = np.where(kd > 0, vn / np.maximum(kd, 1) / s.cut, -1.0)
        if vm:
            vmn = np.nanmin(np.stack(vm), 0) / s.cut; v2 = np.where(np.isfinite(vmn), vmn, -1.0)
        else:
            v2 = np.full(L, -1.0)
        lo3 = max(T - 3, 0); k3 = s.CK[T] - s.CK[lo3]; fr = np.where(k3 > 0, (s.CF[T] - s.CF[lo3]) / np.maximum(k3, 1) / s.cap, np.nan)
        f2 = np.where((kd > 0) & (k3 > 0), fr - f1, np.nan)
        u = np.column_stack([u1, u2, u3, u4, u5, u6, nb(u2, s.dn), nb(u2, s.dn2), nb(u2, s.dn3), nb(u1, s.dn), nb(u1, s.up), f1, f2, v1, v2, nb(v2, s.dn), nb(v2, s.up)]).astype(np.float32)
        return m, x, u
NM = 7
def pick(ts, lo, hi, n=5, gap=72):
    out = []
    for t in ts:
        if t < lo: continue
        if t >= hi or len(out) == n: break
        if out and t - out[-1] < gap: continue
        out.append(int(t))
    return out
def nbrs(G, L):
    r = np.full(L, -1, np.int64)
    for i in range(L):
        nz = G[i].indices
        if len(nz): r[i] = nz[0]
    return r

EV = []; SIM = []; TE = []; SC = []; RK = {}; CHECK = []
ONLINE_CHECK = os.environ.get('TFB_ONLINE_CHECK', '1') == '1'
SPAN = 12 * 288
for pid, name in enumerate(panel_names()):
    P = Pub(name, pid); RK[name] = P.link_rank.copy(); L = len(P.links); z = np.load(f'{PREP}/{name}.npz'); lp = list(META[name]['links']); idx = np.array([lp.index(l) for l in P.links])
    dn = nbrs(P.outgoing, L); up = nbrs(P.incoming, L)
    V = z['train_V'][:, idx].astype(np.float32); Q = z['train_Q'][:, idx].astype(np.float32); E = z['train_E'][:, idx].astype(bool); MA = z['train_Mavail'][:, idx].astype(bool)
    Vmk = np.where(MA, V, np.nan); Qmk = np.where(MA, Q, np.nan)
    n_ev = 0; ev_ts = []; ev_at = {}
    for d in range(NTRAIN):
        s = d * 288; v, q, e = V[s:s + 288], Q[s:s + 288], E[s:s + 288]
        act = ((v <= P.cut[None, :]) & e & np.isfinite(v)).any(1); prev_c = False
        for t in range(HISTORY, 288 - HORIZON):
            c = (np.isfinite(v[t - HISTORY:t]).mean() >= .7 and act[t + 1:t + HORIZON + 1].any() and not act[t - HISTORY:t].any() and not act[t])
            first = c and not prev_c; prev_c = c
            if not first: continue
            T = s + t; origin = START['train'] + pd.Timedelta(minutes=5 * T)
            base = P.features(v[t - HISTORY:t], q[t - HISTORY:t], e[t - HISTORY:t], origin, 'queue_onset')[-L:].astype(np.float32)
            y = (v[t + HORIZON] <= P.cut) & np.isfinite(v[t + HORIZON]); el = e[t + HORIZON] & np.isfinite(v[t + HORIZON])
            ev_at[T] = len(EV); ev_ts.append(T)
            EV.append(dict(panel=name, base=base, y=y, el=el)); n_ev += 1
    # ongoing origins of the train split, as in the official selector, only used to place holes
    qall = (V <= P.cut[None, :]) & E & np.isfinite(V); anyq = qall.any(1); n = len(V); TT = np.arange(n)
    ca = np.concatenate([[0], np.cumsum(anyq.astype(np.int64))]); cnt = lambda a, b: ca[np.clip(b + 1, 0, n)] - ca[np.clip(a, 0, n)]
    ccov = np.concatenate([[0.0], np.cumsum(np.isfinite(V).mean(1))]); cov = np.zeros(n); ok = (TT >= 12) & (TT + 6 < n); cov[ok] = (ccov[TT[ok]] - ccov[TT[ok] - 12]) / 12.0
    ong = ok & anyq & np.r_[False, anyq[:-1]] & (cnt(TT + 1, TT + 6) > 0) & (cov >= .7); og_ts = []
    for t in np.where(ong)[0]:
        q0 = qall[t]; qh = qall[t + 1:t + 7]; uni = (q0[None, :] | qh).sum()
        if uni and (q0[None, :] & qh).sum() / uni <= .9: og_ts.append(int(t))
    n_sim = 0
    for sd in range(0, NTRAIN - 8, 2):
        s0 = sd * 288; H = min(SPAN, n - s0)
        on = pick(ev_ts, s0 + 12, s0 + H); og = pick(og_ts, s0 + 12, s0 + H)
        Vs = Vmk[s0:s0 + H].copy(); Qs = Qmk[s0:s0 + H].copy()
        for T in on + og: Vs[T - s0 + 1:T - s0 + 19] = np.nan; Qs[T - s0 + 1:T - s0 + 19] = np.nan
        for T in on + og: Vs[T - s0 - 12:T - s0] = V[T - 12:T]; Qs[T - s0 - 12:T - s0] = Q[T - 12:T]
        vw = View(Vs, Qs, E[s0:s0 + H], P.cut, P.cap, dn, up)
        for k_, T in enumerate(on):
            m, x, u = vw.feats(T - s0, 0); SIM.append(dict(ev=ev_at[T], k=k_, m=m, x=x, u=u)); n_sim += 1
    # March: calendar onset events outside scored windows, read through the masked layer
    Vm = z['validation_V'][:, idx].astype(np.float32); Qm = z['validation_Q'][:, idx].astype(np.float32); Em = z['validation_E'][:, idx].astype(bool)
    win = pd.read_csv(f'{PREP}/{name}_validation_windows.csv'); bad = np.zeros(len(Vm), bool)
    for t_ in win.t_idx: bad[max(t_ - 12, 0):t_ + 20] = True
    Vm2 = Vm.copy(); Qm2 = Qm.copy(); Vm2[bad] = np.nan; Qm2[bad] = np.nan
    kn = np.isfinite(Vm2) & Em; qq = (np.nan_to_num(Vm2, nan=1e9) <= P.cut[None, :]) & kn; a = qq.any(1); run = 0; te_ts = []; te_ev = {}
    for t0 in range(len(a)):
        if a[t0]:
            if run >= 18 and t0 >= 18 and not bad[t0 - 18:t0 + 1].any():
                T = t0 - 6; origin = START['validation'] + pd.Timedelta(minutes=5 * T)
                base = P.features(Vm2[T - 12:T], Qm2[T - 12:T], Em[T - 12:T], origin, 'queue_onset')[-L:].astype(np.float32)
                pp = .5 * pub_lgb.predict(base, num_threads=4) + .5 * pub_cat.predict_proba(base, thread_count=4)[:, 1]
                te_ts.append(T); te_ev[T] = dict(base=base, y=qq[t0], el=kn[t0], pub=pp)
            run = 0
        else:
            run += 1
    tog = [t for t in range(12, len(a) - 6) if a[t] and a[t - 1] and a[t + 1:t + 7].any() and not bad[t]]
    n_te = 0; nm = len(Vm)
    for sd in range(7, 27):
        s0 = sd * 288; H = min(SPAN, nm - s0)
        on = pick(te_ts, s0 + 12, s0 + H); og = pick(tog, s0 + 12, s0 + H)
        Vs = Vm2[s0:s0 + H].copy(); Qs = Qm2[s0:s0 + H].copy()
        for T in on + og: Vs[T - s0 + 1:T - s0 + 19] = np.nan; Qs[T - s0 + 1:T - s0 + 19] = np.nan
        vw = View(Vs, Qs, Em[s0:s0 + H], P.cut, P.cap, dn, up)
        for k_, T in enumerate(on):
            m, x, u = vw.feats(T - s0, 0); e_ = te_ev[T]
            TE.append(dict(panel=name, sd=sd, k=k_, T=T, base=e_['base'], m=m, x=x, u=u, y=e_['y'], el=e_['el'], pub=e_['pub'])); n_te += 1
    # official scored windows (the prep arrays already carry the histories and the hidden rows). April is read only up to each origin
    Vp = z['private_V'][:, idx].astype(np.float32); Qp = z['private_Q'][:, idx].astype(np.float32); Ep = z['private_E'][:, idx].astype(bool)
    for split, Vs, Qs, Es in (('validation', Vm, Qm, Em), ('private', Vp, Qp, Ep)):
        vw = View(Vs, Qs, Es, P.cut, P.cap, dn, up)
        wn = pd.read_csv(f'{PREP}/{name}_{split}_windows.csv'); wn = wn[wn.condition == 'queue_onset'].sort_values('t_idx')
        for k_, w_ in enumerate(wn.itertuples(index=False)):
            T = int(w_.t_idx); origin = START[split] + pd.Timedelta(minutes=5 * T)
            base = P.features(Vs[T - 12:T], Qs[T - 12:T], Es[T - 12:T], origin, 'queue_onset')[-L:].astype(np.float32)
            m, x, u = vw.feats(T, 0)
            if ONLINE_CHECK:     # the organizers' check: remove every row stamped after T, recompute, compare
                Vc, Qc, Ec = Vs.copy(), Qs.copy(), Es.copy(); Vc[T + 1:] = np.nan; Qc[T + 1:] = np.nan; Ec[T + 1:] = False
                mc, xc, uc = View(Vc, Qc, Ec, P.cut, P.cap, dn, up).feats(T, 0)
                bc = P.features(Vc[T - 12:T], Qc[T - 12:T], Ec[T - 12:T], origin, 'queue_onset')[-L:].astype(np.float32)
                diff = max(float(np.nanmax(np.abs(a_.astype(np.float64) - b_.astype(np.float64)), initial=0.0)) for a_, b_ in ((m, mc), (x, xc), (u, uc), (base, bc)))
                nanpat = all(np.array_equal(np.isnan(a_), np.isnan(b_)) for a_, b_ in ((m, mc), (x, xc), (u, uc), (base, bc)))
                CHECK.append(dict(split=split, window_id=w_.window_id, max_abs_diff=diff, same=bool(nanpat and all(np.allclose(a_, b_, rtol=1e-5, atol=1e-6, equal_nan=True) for a_, b_ in ((m, mc), (x, xc), (u, uc), (base, bc))))))     # tolerance: float sums differ in the last bits with memory alignment
            SC.append(dict(panel=name, split=split, k=k_, window_id=w_.window_id, origin=str(w_.forecast_origin), links=list(P.links), base=base, m=m, x=x, u=u, pub=.5 * pub_lgb.predict(base, num_threads=4) + .5 * pub_cat.predict_proba(base, thread_count=4)[:, 1]))
    log(name, 'train events', n_ev, 'pseudo-month rows', n_sim, 'March events', len(te_ts), 'March rows', n_te)
    del V, Q, E, MA, Vmk, Qmk
log('train events', len(EV), 'pseudo-month rows', len(SIM), 'March rows', len(TE), 'scored', len(SC))
if ONLINE_CHECK:
    ck = pd.DataFrame(CHECK); ck.to_csv(_os.path.join(TFB_OUT, 'online_rule_check_onset.csv'), index=False)
    log('online-rule check (onset): windows', len(ck), 'features unchanged with rows after T removed', int(ck.same.sum()), 'max abs difference', float(ck.max_abs_diff.max())); assert ck.same.all(), ck[~ck.same]

PARAMS = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': .04, 'num_leaves': 15, 'min_data_in_leaf': 100, 'lambda_l2': 10, 'feature_fraction': .9, 'num_threads': 4, 'verbosity': -1, 'seed': 20261004}
NB = EV[0]['base'].shape[1]
MP = NB - 1 - 1 - 4 * 6                         # milepost column
def cols(drop_id):
    keep = np.ones(NB, bool)
    if drop_id == 'identity': keep[[0, 1, MP]] = False       # panel number, link index, milepost: pure location memory
    return np.flatnonzero(keep)
def XY_S(k, extra):
    Xs, ys = [], []
    for r in SIM:
        e = EV[r['ev']]; x = np.column_stack([e['base'][:, k], r['m']] + [r[n_] for n_ in extra]); Xs.append(x[e['el']]); ys.append(e['y'][e['el']])
    return np.concatenate(Xs), np.concatenate(ys).astype(np.float32)
def iou(sel, e):
    t = e['y'] & e['el']; p = sel & e['el']; u = (t | p).sum()
    return 1.0 if u == 0 else (t & p).sum() / u
def dec(p, thr):
    s = p >= thr
    if not s.any(): s[np.argmax(p)] = True
    return s
def eiou(p):
    o = np.argsort(-p); sp = p[o]; cs = np.cumsum(sp); k = np.arange(1, len(p) + 1)
    s = np.zeros(len(p), bool); s[o[:int(np.argmax(cs / (k + (sp.sum() - cs)))) + 1]] = True
    return s
def report(name, P_):
    d = pd.DataFrame([dict(panel=e['panel'], k=e['k'], i55=iou(dec(p, .55), e), i45=iou(dec(p, .45), e), i35=iou(dec(p, .35), e), ie=iou(eiou(p), e)) for e, p in zip(TE, P_)])
    r = dict(name=name, i55=d.groupby('panel').i55.mean().mean(), i45=d.groupby('panel').i45.mean().mean(), i35=d.groupby('panel').i35.mean().mean(), ie=d.groupby('panel').ie.mean().mean(), **{f'ie_k{k_}': g.groupby('panel').ie.mean().mean() for k_, g in d.groupby('k')})
    log({k_: (round(v, 4) if not isinstance(v, str) else v) for k_, v in r.items()})
    return r
RES = []; RES.append(report('released public', [e['pub'] for e in TE]))
Pt = {}; Sc = {}
for di in ('none', 'identity'):
    k = cols(di); X, y = XY_S(k, ['x', 'u']); log('rows', X.shape)
    tofeat = lambda e: np.column_stack([e['base'][:, k], e['m'], e['x'], e['u']])
    ds = lgb.Dataset(X, label=y, free_raw_data=False)
    b = lgb.train(PARAMS, ds, num_boost_round=1500); log('fitted D', di)
    Pt[('D', di)] = [b.predict(tofeat(e), num_threads=4) for e in TE]; Sc[('D', di)] = [b.predict(tofeat(e), num_threads=4) for e in SC]; del b
    RES.append(report(f'D {di}', Pt[('D', di)]))
    pt, sc = [], []
    for sd_ in (11, 22, 33):
        m_ = lgb.train(dict(PARAMS, seed=sd_, feature_fraction=.7, bagging_fraction=.8, bagging_freq=1), ds, num_boost_round=1500); log('fitted bag', di, sd_)
        pt.append([m_.predict(tofeat(e), num_threads=4) for e in TE]); sc.append([m_.predict(tofeat(e), num_threads=4) for e in SC]); del m_
    Pt[('Bag', di)] = [np.mean([p_[i_] for p_ in pt], 0) for i_ in range(len(TE))]; Sc[('Bag', di)] = [np.mean([p_[i_] for p_ in sc], 0) for i_ in range(len(SC))]
    RES.append(report(f'Bag {di}', Pt[('Bag', di)]))
    c = CatBoostClassifier(iterations=1500, depth=7, learning_rate=.06, l2_leaf_reg=10, task_type='CPU', thread_count=4, random_seed=20261005, loss_function='Logloss', verbose=False, allow_writing_files=False)
    c.fit(X, y); log('fitted cat', di)
    Pt[('Cat', di)] = [c.predict_proba(tofeat(e))[:, 1] for e in TE]; Sc[('Cat', di)] = [c.predict_proba(tofeat(e))[:, 1] for e in SC]; del c, X, y, ds
    RES.append(report(f'Cat {di}', Pt[('Cat', di)]))
def both(tag): return [.5 * (p + q) for p, q in zip(Pt[(tag, 'none')], Pt[(tag, 'identity')])]
for tag in ('D', 'Bag', 'Cat'):
    RES.append(report(f'{tag} both', both(tag)))
bl = [.5 * (p + q) for p, q in zip(both('Bag'), both('Cat'))]
RES.append(report('Bag + Cat', bl)); RES.append(report('Bag + Cat, k0 .6 with public', [(.6 * p + .4 * e['pub']) if e['k'] == 0 else p for p, e in zip(bl, TE)]))
RES.append(report('Bag, k0 .6 with public', [(.6 * p + .4 * e['pub']) if e['k'] == 0 else p for p, e in zip(both('Bag'), TE)]))
# where the misses fall: link distance from predicted to true cells
rows = []
for e, p in zip(TE, both('D')):
    t = np.flatnonzero(e['y'] & e['el']); pr = np.flatnonzero(eiou(p) & e['el']); rk = RK[e['panel']]
    if not len(t) or not len(pr): continue
    dist = np.abs(rk[pr][:, None] - rk[t][None, :]); rows.append(dict(panel=e['panel'], k=e['k'], ie=iou(eiou(p), e), nt=len(t), np_=len(pr), dmin=int(dist.min()), dmax=int(dist.min(1).max()), hour=(e['T'] % 288) / 12.0, tp=float(p[t].max()), pm=float(p.max())))
M = pd.DataFrame(rows); miss = M[M.ie == 0]
log('misses: share', round((M.ie == 0).mean(), 3), 'distance of nearest predicted cell to a true cell', miss.dmin.value_counts().sort_index().to_dict(), 'partial windows', int(((M.ie > 0) & (M.ie < 1)).sum()), 'perfect', int((M.ie == 1).sum()))
log('misses by corridor ' + str(miss.groupby('panel').size().to_dict()) + ' top probability on a true cell in misses ' + str(round(miss.tp.mean(), 3)) + ' vs top overall ' + str(round(miss.pm.mean(), 3)))
log('I5_S and I10_W events (unique)\n' + M[M.panel.isin(['D12_I5_S', 'D7_I10_W'])].drop_duplicates(['panel', 'hour', 'nt']).round(2).head(40).to_string(index=False))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 100)
log('results\n' + pd.DataFrame(RES).round(4).to_string(index=False))
key = pd.read_csv(f'{DATA}/submission_key.csv', dtype=str, usecols=['submission_id', 'task', 'window_id', 'timestamp', 'link_id']); key = key[key.task == 'queue'].drop(columns='task')
rows = []
for i, w in enumerate(SC):
    f = pd.DataFrame({'window_id': w['window_id'], 'link_id': w['links'], 'k': w['k'], 'pub': w['pub'], 'timestamp': (pd.Timestamp(w['origin']) + pd.Timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ')})
    for (tag, di), v in Sc.items(): f[f'{tag}_{di}'] = v[i]
    rows.append(f)
pr = pd.concat(rows, ignore_index=True); m_ = key.merge(pr, on=['window_id', 'timestamp', 'link_id'], how='inner'); assert len(m_) == len(pr), (len(m_), len(pr))
m_.to_csv(_os.path.join(TFB_OUT, 'onset_probs.csv'), index=False); log('written', len(m_), 'rows; windows', m_.window_id.nunique())
log('done')
