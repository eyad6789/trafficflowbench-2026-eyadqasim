"""Stage 05 - Task 2 ongoing. LightGBM (lr .03, 127 leaves, 1400 rounds) on queue-geometry and shock-speed features of the 60-minute
history and the row at T, with no profiles, no link or corridor index and official capacity, trained on train-split pseudo-months that
copy the official selector (6 h between any two windows, 5 per condition, the 90 minutes after each origin hidden). Two models, with and
without month-to-date evidence channels (rows stamped before T only); the submission uses their mean. Writes ongoing_probs.csv, prints the
March pseudo-month check (public ongoing model of anubisyy, Apache-2.0, as comparison) and runs the online-rule check."""
MODE = "final"
# ---- paths: Kaggle by default; run_all.py points these at local folders -------------------------------------
import os as _os, glob as _glob
TFB_ROOTS = [r for r in _os.environ.get("TFB_INPUT", "/kaggle/input").split(_os.pathsep) if r]
TFB_OUT = _os.environ.get("TFB_OUT", "/kaggle/working"); _os.makedirs(TFB_OUT, exist_ok=True)
def G(pat):
    """every file matching pat under the input roots (recursive), sorted"""
    return sorted(set(p for r in TFB_ROOTS for p in _glob.glob(_os.path.join(r, "**", pat), recursive=True)))
# ----------------------------------------------------------------------------------------------------------------
import os, sys, glob, json, time, math, gc, warnings, shutil
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F_nn
import lightgbm as lgb
warnings.filterwarnings("ignore")
T0 = time.time()
def log(*a):
    print(*a, f"[{time.time()-T0:.0f}s]", flush=True)

CFG = dict(seed=2026, folds=4, extra_per_panel=5000, epochs=16, bs=48, lr=2e-3, wd=1e-4, width=96, sel_w=2.0,
           do_onset=False, onset_epochs=14)   # SEED_PATCH
DATA = G("kaggle_public")[0]
PREP = os.path.dirname(G("prep/meta.json")[0])
META = json.load(open(f"{PREP}/meta.json"))
PANELS = META["panels"]
SPL = {s: (v[0], int(v[1])) for s, v in META["splits"].items()}     # first day and number of days of each split, read from the data by stage 01
OUT = TFB_OUT
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(CFG["seed"]); RNG = np.random.default_rng(CFG["seed"])
log("DATA", DATA, "PREP", PREP, "device", DEV, torch.__version__)


def ffill_t(X):
    w, T, L = X.shape
    ok = np.isfinite(X)
    idx = np.where(ok, np.arange(T)[None, :, None], -1)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.take_along_axis(X, np.maximum(idx, 0), axis=1)
    out[idx < 0] = np.nan
    return out

def bfill_t(X):
    return ffill_t(X[:, ::-1])[:, ::-1]

def fill_space(X):
    L = X.shape[-1]
    idx = np.where(np.isfinite(X), np.arange(L), -1)
    np.maximum.accumulate(idx, axis=-1, out=idx)
    fw = np.take_along_axis(X, np.maximum(idx, 0), axis=-1); fw[idx < 0] = np.nan
    Xr = X[..., ::-1]
    idx2 = np.where(np.isfinite(Xr), np.arange(L), -1)
    np.maximum.accumulate(idx2, axis=-1, out=idx2)
    bw = np.take_along_axis(Xr, np.maximum(idx2, 0), axis=-1); bw[idx2 < 0] = np.nan
    bw = bw[..., ::-1]
    both = np.isfinite(fw) & np.isfinite(bw)
    return np.where(both, 0.5 * (fw + bw), np.where(np.isfinite(fw), fw, bw))

def shift(X, d):
    """value at link i+d (d>0 downstream), edge padded"""
    L = X.shape[-1]
    return X[..., np.clip(np.arange(L) + d, 0, L - 1)]

def win_min(X, lo, hi):
    out = shift(X, lo).copy()
    for d in range(lo + 1, hi + 1):
        np.minimum(out, shift(X, d), out=out)
    return out

def win_sum(X, lo, hi):
    out = np.zeros(X.shape, np.float32)
    L = X.shape[-1]
    for d in range(lo, hi + 1):
        idx = np.arange(L) + d
        ok = (idx >= 0) & (idx < L)
        out[..., ok] += X[..., idx[ok]]
    return out

def nearest_dn(q):
    L = q.shape[-1]
    idx = np.where(q, np.arange(L), L)
    return np.minimum.accumulate(idx[..., ::-1], axis=-1)[..., ::-1]

def nearest_up(q):
    L = q.shape[-1]
    idx = np.where(q, np.arange(L), -1)
    return np.maximum.accumulate(idx, axis=-1)

def rank01(X):
    return (X.argsort(-1).argsort(-1) / max(X.shape[-1] - 1, 1)).astype(np.float32)



class Panel:
    def __init__(self, name, pid):
        z = dict(np.load(f"{PREP}/{name}.npz"))
        m = META[name]
        self.name, self.pid = name, pid
        links = list(m["links"]); L = self.L = len(links)
        # travel direction: does traffic flow towards higher or lower order index? (topology outgoing links)
        tp = pd.read_csv(f"{DATA}/corridors/{name}/network/lwr_mainline_topology.csv", dtype=str)
        outs = {l: set(str(o).split(";")) for l, o in zip(tp.link_id.astype(str), tp.outgoing_link_ids)}
        fwd = sum(b in outs.get(a, ()) for a, b in zip(links[:-1], links[1:]))
        bwd = sum(a in outs.get(b, ()) for a, b in zip(links[:-1], links[1:]))
        mf = sorted(glob.glob(f"{DATA}/corridors/{name}/train/mainline_states/**/*.parquet", recursive=True))[0]
        import pyarrow.parquet as _pq
        mp = _pq.ParquetFile(mf).read(columns=["link_id", "milepost"]).to_pandas().drop_duplicates("link_id")
        mpv = mp.set_index(mp.link_id.astype(str)).milepost.reindex(links).to_numpy(float)
        dmp = np.diff(mpv)
        self.reversed = bwd > fwd
        if self.reversed:
            links = links[::-1]
            for k in list(z):
                if k in ("vf", "lanes", "cap", "length", "kc", "kj", "n_on_topo", "n_off_topo"):
                    z[k] = z[k][::-1].copy()
                elif k.endswith(("_V", "_Q", "_O", "_E", "_Mavail")):
                    z[k] = z[k][:, ::-1].copy()
            z["ramp_link"] = np.where(z["ramp_link"] >= 0, L - 1 - z["ramp_link"], -1)
            mpv = mpv[::-1]; dmp = np.diff(mpv)
        self.direction_info = dict(fwd=int(fwd), bwd=int(bwd), reversed=bool(self.reversed),
                                   milepost_up=float(np.mean(dmp > 0)), milepost_down=float(np.mean(dmp < 0)))
        self.milepost = mpv.astype(np.float32)
        self.links = links
        self.vf = z["vf"].astype(np.float32); self.vcut = (0.6 * self.vf).astype(np.float32)
        self.lanes = z["lanes"].astype(np.float32); self.cap = z["cap"].astype(np.float32)
        self.length = z["length"].astype(np.float32)
        kc = z["kc"].astype(np.float32)
        self.kc = np.where(np.isfinite(kc) & (kc > 0), kc, self.cap / self.vf).astype(np.float32)
        self.cum = (np.concatenate([[0.0], np.cumsum(self.length)[:-1]]) + 0.5 * self.length).astype(np.float32)
        rl = z["ramp_link"]; rt = np.array(m["ramp_type"])
        self.ramp_types = {str(k): int(v) for k, v in pd.Series(rt).value_counts().items()}
        is_on = np.isin(rt, ["OR", "ON", "on", "onramp", "on_ramp"])
        self.A_on = np.zeros((len(rl), L), np.float32); self.A_off = np.zeros((len(rl), L), np.float32)
        for k, li in enumerate(rl):
            if li >= 0:
                (self.A_on if is_on[k] else self.A_off)[k, li] = 1.0
        self.n_on = z["n_on_topo"].astype(np.float32); self.n_off = z["n_off_topo"].astype(np.float32)
        self.S = {}
        for s, (start, nd) in SPL.items():
            d = dict(V=z[f"{s}_V"].astype(np.float32), Q=z[f"{s}_Q"].astype(np.float32), O=z[f"{s}_O"].astype(np.float32),
                     E=z[f"{s}_E"].astype(bool), R=z[f"{s}_R"].astype(np.float32))
            if s == "train":
                d["M"] = z["train_Mavail"].astype(bool)
            t0 = pd.Timestamp(start, tz="UTC")
            d["t0"] = t0; d["n"] = nd * 288
            d["dow"] = np.repeat((t0.dayofweek + np.arange(nd)) % 7, 288).astype(np.int64)
            d["win"] = pd.read_csv(f"{PREP}/{name}_{s}_windows.csv")
            self.S[s] = d
        tr = self.S["train"]
        qq = np.where(tr["E"], tr["Q"], np.nan)
        cap_emp = np.nanpercentile(qq, 99.5, axis=0)
        self.cap_emp = np.where(np.isfinite(cap_emp) & (cap_emp > 100), cap_emp, self.cap).astype(np.float32)
        del qq
        self.known = tr["E"] & np.isfinite(tr["V"])
        self.lab = (tr["V"] <= self.vcut[None, :]) & self.known
        self._profiles()
        log(f"panel {name}: L={L} direction={self.direction_info} ramps={self.ramp_types} queued-rate(train,known)={self.lab.sum()/self.known.sum():.4f}")

    def _profiles(self):
        tr = self.S["train"]; n, L = tr["V"].shape; nd = n // 288
        r = tr["V"] / self.vf[None, :]
        rk = np.where(self.known, r, 0.0).astype(np.float32)
        lab = self.lab.astype(np.float32); kn = self.known.astype(np.float32)
        def sm3(a):   # sum over t-1, t, t+1 on the absolute timeline
            o = a.copy(); o[1:] += a[:-1]; o[:-1] += a[1:]; return o
        self.own_q3 = sm3(lab).astype(np.float16)
        self.own_k3 = sm3(kn).astype(np.float16)
        self.own_r3 = sm3(rk).astype(np.float32)
        dows = tr["dow"][::288]
        SQ = np.zeros((7, 288, L), np.float32); SK = SQ.copy(); SR = SQ.copy()
        for dw in range(7):
            ds = np.where(dows == dw)[0]
            if len(ds):
                SQ[dw] = lab.reshape(nd, 288, L)[ds].sum(0); SK[dw] = kn.reshape(nd, 288, L)[ds].sum(0)
                SR[dw] = rk.reshape(nd, 288, L)[ds].sum(0)
        sm = lambda a: a + np.roll(a, 1, axis=1) + np.roll(a, -1, axis=1)
        self.SQ3, self.SK3, self.SR3 = sm(SQ), sm(SK), sm(SR)

    def prof(self, split, Ts, h, which):
        """historical P(queued) ('q') or mean speed ratio ('r') at time T+h for every link; leave-own-cell-out on train"""
        d = self.S[split]
        t = np.clip(np.asarray(Ts, np.int64) + h, 0, d["n"] - 1)
        dw = d["dow"][t]; sl = t % 288
        num = (self.SQ3 if which == "q" else self.SR3)[dw, sl]
        den = self.SK3[dw, sl]
        if split == "train":
            num = num - (self.own_q3 if which == "q" else self.own_r3)[t].astype(np.float32)
            den = den - self.own_k3[t].astype(np.float32)
        out = num / np.maximum(den, 1.0)
        if which == "r":
            out = np.where(den >= 1, out, 1.0)
        return out.astype(np.float32)


# ----------------------------------------------------------------------------------------------------------

def imputed_queue(P):
    V = P.S["train"]["V"]
    r = V / P.vf[None, :]
    prev = pd.DataFrame(r).ffill(limit=2).to_numpy(); nxt = pd.DataFrame(r).bfill(limit=2).to_numpy()
    rf = np.where(np.isfinite(r), r, np.where(np.isfinite(prev) & np.isfinite(nxt), 0.5 * (prev + nxt), np.nan))
    return np.nan_to_num(rf, nan=1.0) <= 0.6

def candidates(P):
    n = P.S["train"]["n"]
    qi = imputed_queue(P)
    anyq = qi.any(1)
    E = P.S["train"]["E"]
    T = np.arange(n)
    ok = (T >= 12) & (T + 6 < n)
    ce = np.concatenate([[0.0], np.cumsum(E.mean(1))])
    cov = np.zeros(n, np.float32)
    cov[ok] = (ce[T[ok]] - ce[T[ok] - 12]) / 12.0
    ca = np.concatenate([[0], np.cumsum(anyq.astype(np.int64))])
    cnt = lambda a, b: ca[np.clip(b + 1, 0, n)] - ca[np.clip(a, 0, n)]
    onset = ok & (cnt(T - 12, T + 5) == 0) & np.r_[anyq[6:], np.zeros(6, bool)] & (cov >= 0.7)
    ong = ok & anyq & np.r_[False, anyq[:-1]] & (cnt(T + 1, T + 6) > 0) & (cov >= 0.7)
    pers_imp = np.ones(n, np.float32); pers_kn = np.ones(n, np.float32)
    idx = np.where(ong)[0]
    for a in range(0, len(idx), 2048):
        tt = idx[a:a + 2048]
        rows = tt[:, None] + np.arange(1, 7)[None, :]
        q0 = qi[tt][:, None, :]
        qh = qi[rows]
        inter = (q0 & qh).sum((1, 2)); uni = (q0 | qh).sum((1, 2))
        pers_imp[tt] = np.where(uni > 0, inter / np.maximum(uni, 1), 1.0)
        lab = P.lab[rows]; kn = P.known[rows]
        pr = q0 & kn
        inter = (pr & lab).sum((1, 2)); uni = (pr | lab).sum((1, 2))
        pers_kn[tt] = np.where(uni > 0, inter / np.maximum(uni, 1), 1.0)
    return onset, ong, pers_imp, pers_kn


def iou(pred, lab, kn):
    p = pred & kn; t = lab & kn
    u = (p | t).sum()
    return 1.0 if u == 0 else float((p & t).sum() / u)

def decode_topk(p, cap_k=None):
    """expected-IoU top-k (ratio-of-expectations approximation), k >= 1"""
    order = np.argsort(-p)
    ps = p[order]
    cs = np.cumsum(ps); tot = ps.sum()
    k = np.arange(1, len(p) + 1)
    e = cs / (k + (tot - cs))
    if cap_k:
        e = e[:cap_k]
    kb = int(np.argmax(e)) + 1
    sel = np.zeros(len(p), bool); sel[order[:kb]] = True
    return sel

def decode_thr(p, thr):
    sel = p >= thr
    if not sel.any():
        sel = np.zeros(p.shape, bool); sel.flat[int(np.argmax(p))] = True
    return sel

# ==========================================================================================================

def greedy(onset_ok, ong_ok, s, e, spacing=72):
    """official-like selector: chronological scan, every selected window (either condition) >= 360 min after
    the previous selected one, at most 5 windows per condition"""
    sel_on, sel_og, last = [], [], -10 ** 9
    for T in np.flatnonzero((onset_ok | ong_ok)[s + 12:e]) + s + 12:
        T = int(T)
        if T - last < spacing:
            continue
        if onset_ok[T] and len(sel_on) < 5:
            sel_on.append(T); last = T
        elif ong_ok[T] and len(sel_og) < 5:
            sel_og.append(T); last = T
        if len(sel_on) == 5 and len(sel_og) == 5:
            break
    return sel_on, sel_og



def make_inputs(P, split, Ts):
    d = P.S[split]
    Ts = np.asarray(Ts, np.int64)
    rows = Ts[:, None] + np.arange(-12, 1)[None, :]
    V = d["V"][rows]; Q = d["Q"][rows]; O = d["O"][rows]; R = d["R"][rows]
    if split == "train":            # the row at T is published Task-1 masked: mimic with the train masked layer
        M = d["M"][Ts]
        for X in (V, Q, O):
            XT = X[:, 12]; XT[~M] = np.nan
    return V, Q, O, R


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


PAN = [Panel(name, pid) for pid, name in enumerate(PANELS)]
for P in PAN:
    P.cap_emp = P.cap.copy()
    del P.SQ3, P.SK3, P.SR3
SRC = _os.path.join(TFB_OUT, "src"); os.makedirs(SRC, exist_ok=True)
for f_ in ("config.py", "queue_ongoing.py"):
    shutil.copy([c for c in G(f_) if "public_pipeline" in c or "catboost" in c][0], f"{SRC}/{f_}")     # public pipeline code (anubisyy, Apache-2.0), used only for the comparison model
os.environ.update(TFB_DATA_ROOT=DATA, TFB_OUTPUT_ROOT=_os.path.join(TFB_OUT, "tmp"), TFB_THREADS="4"); sys.path.insert(0, SRC)
from queue_ongoing import Panel as PubOng, panel_names
ASSETS = os.path.dirname(G("ongoing_lgb.txt")[0])
pub_model = lgb.Booster(model_file=f"{ASSETS}/ongoing_lgb.txt"); PUBN = panel_names()
def pub_pred(P, Pb, idx, inv, split, T):
    d = P.S[split]; origin = d["t0"] + pd.Timedelta(minutes=5 * T)
    return pub_model.predict(Pb.features(d["V"][T - 12:T][:, idx], d["Q"][T - 12:T][:, idx], d["E"][T - 12:T][:, idx], origin, "queue_ongoing"), num_threads=4).reshape(6, -1)[:, inv].astype(np.float32)

NAMES = None
def feats_og(P, split, Ts, Mtd):
    """(w * L * 6, n features) float32"""
    global NAMES
    Ts = np.asarray(Ts, np.int64)
    V, Q, O, R = make_inputs(P, split, Ts)
    w, _, L = V.shape
    vf = P.vf[None, None, :]
    F = {}
    F["obsT"] = np.isfinite(V[:, 12]).astype(np.float32)
    F["obs_frac"] = np.isfinite(V).mean(1).astype(np.float32)
    rf = fill_space(bfill_t(ffill_t(V / vf))); rf = np.where(np.isfinite(rf), rf, 1.0).astype(np.float32)
    qf = fill_space(bfill_t(ffill_t(Q))); qf = np.where(np.isfinite(qf), qf, 0.5 * P.cap[None, None, :]).astype(np.float32)
    of = fill_space(bfill_t(ffill_t(O))); of = np.where(np.isfinite(of), of, 0.05).astype(np.float32)
    rT = rf[:, 12]; F["rT"] = rT
    for k, row in (("r5", 11), ("r10", 10), ("r15", 9), ("r30", 6), ("r60", 0)):
        F[k] = rf[:, row]
    F["d5"] = rT - rf[:, 11]; F["d15"] = rT - rf[:, 9]; F["d30"] = rT - rf[:, 6]; F["d60"] = rT - rf[:, 0]
    x = np.arange(7, dtype=np.float32) - 3.0
    F["slope30"] = (rf[:, 6:13] * x[None, :, None]).sum(1) / float((x ** 2).sum())
    F["rproj"] = rT + 6.0 * F["slope30"]; F["rproj15"] = rT + 2.0 * F["d15"]
    F["rmin"] = rf.min(1); F["rmax"] = rf.max(1); F["rstd"] = rf.std(1)
    capr = P.cap[None, :]
    F["fT"] = qf[:, 12] / capr; F["f3"] = qf[:, 10:13].mean(1) / capr; F["fd30"] = (qf[:, 12] - qf[:, 6]) / capr
    F["fe_proj"] = (qf[:, 10:13].mean(1) + (qf[:, 10:13].mean(1) - qf[:, 4:7].mean(1))) / capr
    F["fe_max12"] = qf.max(1) / capr
    k = qf / np.maximum(rf * vf, 3.0)
    F["kT"] = k[:, 12] / P.kc[None, :]; F["kd15"] = (k[:, 12] - k[:, 9]) / P.kc[None, :]
    F["oT"] = of[:, 12]; F["od15"] = of[:, 12] - of[:, 9]
    for dd in (-3, -2, -1, 1, 2, 3):
        F[f"rT{dd:+d}"] = shift(rT, dd); F[f"rproj{dd:+d}"] = shift(F["rproj"], dd)
    F["rmin_up5"] = win_min(rT, -5, -1); F["rmin_dn5"] = win_min(rT, 1, 5)
    F["rmin_up15"] = win_min(rT, -15, -6); F["rmin_dn15"] = win_min(rT, 6, 15)
    F["d15_up1"] = shift(F["d15"], -1); F["d15_dn1"] = shift(F["d15"], 1)
    F["kT_dn1"] = shift(F["kT"], 1); F["kT_up1"] = shift(F["kT"], -1)
    Rl = np.nan_to_num(ffill_t(R)[:, 12], nan=0.0); R30 = np.nan_to_num(ffill_t(R)[:, 6], nan=0.0)
    on = Rl @ P.A_on; off = Rl @ P.A_off; on30 = R30 @ P.A_on
    F["onT"] = on / capr; F["offT"] = off / capr; F["on_d30"] = (on - on30) / capr
    F["on_up5"] = win_sum(on, -5, 0) / capr; F["off_dn5"] = win_sum(off, 0, 5) / capr
    has_on = (P.A_on.sum(0) > 0).astype(np.float32)
    st = {"vf": P.vf, "lanes": P.lanes, "lane_dn": shift(P.lanes, 1) - P.lanes, "lane_up": P.lanes - shift(P.lanes, -1), "capl": P.cap / P.lanes, "len": P.length, "n_on": P.n_on, "n_off": P.n_off,
          "has_on": has_on, "has_off": (P.A_off.sum(0) > 0).astype(np.float32), "on_near": win_sum(has_on[None, :], -3, 3)[0]}
    for kk, v in st.items():
        F[kk] = np.broadcast_to(np.asarray(v, np.float32)[None, :], (w, L))
    slot = Ts % 288; dow = P.S[split]["dow"][Ts]
    F["tod"] = np.broadcast_to((slot / 12.0).astype(np.float32)[:, None], (w, L))
    F["wk"] = np.broadcast_to((dow >= 5).astype(np.float32)[:, None], (w, L))
    qh = rf <= 0.6; qT = qh[:, 12]
    F["qT"] = qT.astype(np.float32); F["q5"] = qh[:, 11].astype(np.float32); F["q10"] = qh[:, 10].astype(np.float32)
    F["qcnt"] = qh.sum(1).astype(np.float32)
    age = np.zeros((w, L), np.float32); run = np.ones((w, L), bool)
    for t in range(12, -1, -1):
        run &= qh[:, t]; age += run
    F["qage"] = age
    last = np.full((w, L), 13.0, np.float32)
    for t in range(13):
        last = np.where(qh[:, t], 12 - t, last)
    F["since_q"] = last
    for dd in (-3, -2, -1, 1, 2, 3):
        F[f"qT{dd:+d}"] = shift(F["qT"], dd)
    F["qsum_up10"] = win_sum(F["qT"], -10, -1); F["qsum_dn10"] = win_sum(F["qT"], 1, 10)
    cumx = np.concatenate([P.cum, [P.cum[-1] + 50.0]]).astype(np.float32)
    for tag, row in (("", 12), ("_m15", 9), ("_m30", 6)):
        q = qh[:, row]
        jd = nearest_dn(q); ju = nearest_up(q)
        ddn = np.where(jd < L, cumx[np.minimum(jd, L)] - P.cum[None, :], 50.0)
        dup = np.where(ju >= 0, P.cum[None, :] - P.cum[np.maximum(ju, 0)], 50.0)
        F["ddn" + tag] = np.minimum(ddn, 50.0).astype(np.float32); F["dup" + tag] = np.minimum(dup, 50.0).astype(np.float32)
        F["nq" + tag] = np.broadcast_to(q.mean(1, keepdims=True).astype(np.float32), (w, L))
    F["ddn_chg15"] = F["ddn"] - F["ddn_m15"]; F["dup_chg15"] = F["dup"] - F["dup_m15"]
    F["ddn_chg30"] = F["ddn"] - F["ddn_m30"]; F["dup_chg30"] = F["dup"] - F["dup_m30"]
    F["nq_chg15"] = F["nq"] - F["nq_m15"]; F["nq_chg30"] = F["nq"] - F["nq_m30"]; F["nq_cnt"] = F["nq"] * L
    hh = np.arange(1, 7, dtype=np.float32)[None, None, :]
    F["ddn_proj15"] = F["ddn"][..., None] + (F["ddn_chg15"][..., None] / 3.0) * hh
    F["ddn_proj30"] = F["ddn"][..., None] + (F["ddn_chg30"][..., None] / 6.0) * hh
    F["dup_proj15"] = F["dup"][..., None] + (F["dup_chg15"][..., None] / 3.0) * hh
    F["rproj_h"] = rT[..., None] + F["slope30"][..., None] * hh
    wi = np.arange(w)[:, None]; qT_ = qf[:, 12]; kv = qT_ / np.maximum(rT * P.vf[None, :], 3.0)
    def shock(j):
        j = np.clip(j, 0, L - 1); u1, u2, d1 = np.maximum(j - 1, 0), np.maximum(j - 2, 0), np.minimum(j + 1, L - 1)
        q_up = 0.5 * (qT_[wi, u1] + qT_[wi, u2]); k_up = 0.5 * (kv[wi, u1] + kv[wi, u2]); q_in = 0.5 * (qT_[wi, j] + qT_[wi, d1]); k_in = 0.5 * (kv[wi, j] + kv[wi, d1])
        dk = k_up - k_in
        return np.clip(np.where(np.abs(dk) > 1.0, (q_up - q_in) / np.where(np.abs(dk) > 1.0, dk, 1.0), 0.0), -40, 40).astype(np.float32)
    q12 = qh[:, 12]; jd = nearest_dn(q12); w_dn = np.where(jd < L, shock(jd), 0.0); F["w_tail"] = w_dn
    t_reach = np.where((jd < L) & (w_dn < -0.5), F["ddn"] / np.maximum(-w_dn, 0.5) * 12.0, 99.0).astype(np.float32); F["t_reach"] = np.minimum(t_reach, 99.0)
    F["reach_h"] = hh - F["t_reach"][..., None]
    nu = nearest_up(~q12); tail_in = np.clip(nu + 1, 0, L - 1); w_in = np.where(q12, shock(tail_in), 0.0); F["w_seg_tail"] = w_in
    dseg = P.cum[None, :] - P.cum[tail_in]; t_clear = np.where(q12 & (w_in > 0.5), dseg / np.maximum(w_in, 0.5) * 12.0, 99.0).astype(np.float32); F["t_clear"] = np.minimum(t_clear, 99.0)
    F["clear_h"] = hh - F["t_clear"][..., None]
    for thr_, tg in ((.7, "70"), (.8, "80")):                       # slow links next to a queue, before they cross the cut
        q_ = rT <= thr_; jd_ = nearest_dn(q_); ju_ = nearest_up(q_)
        F["x2_ddn" + tg] = np.minimum(np.where(jd_ < L, cumx[np.minimum(jd_, L)] - P.cum[None, :], 50.0), 50.0).astype(np.float32)
        F["x2_dup" + tg] = np.minimum(np.where(ju_ >= 0, P.cum[None, :] - P.cum[np.maximum(ju_, 0)], 50.0), 50.0).astype(np.float32)
    up_free = nearest_up(~q12); dn_free = nearest_dn(~q12); blk = np.where(q12, dn_free - up_free - 1, 0)    # the queued block around a link
    F["x2_blk"] = blk.astype(np.float32); F["x2_blk_head"] = np.where(q12, dn_free - 1 - np.arange(L)[None, :], -1).astype(np.float32); F["x2_blk_tail"] = np.where(q12, np.arange(L)[None, :] - up_free - 1, -1).astype(np.float32)
    jd12 = nearest_dn(q12); F["x2_blk_dn"] = np.where(jd12 < L, blk[wi, np.minimum(jd12, L - 1)], 0).astype(np.float32)
    for dd in (-1, 1):
        F[f"x2_fT{dd:+d}"] = shift(F["fT"], dd); F[f"x2_oT{dd:+d}"] = shift(F["oT"], dd)
    f3_ = qf[:, 10:13].mean(1)
    F["x2_fdiff"] = (shift(qf[:, 12], -1) - shift(qf[:, 12], 1)) / capr; F["x2_fdiff3"] = (shift(f3_, -1) - shift(f3_, 1)) / capr      # vehicles piling up between neighbours
    for dd in (-6, -5, -4, 4, 5, 6):
        F[f"x2_rT{dd:+d}"] = shift(rT, dd); F[f"x2_qT{dd:+d}"] = shift(F["qT"], dd)
    F["x2_d15-2"] = shift(F["d15"], -2); F["x2_d15+2"] = shift(F["d15"], 2)
    keys = list(F.keys()); n = w * L * 6; nm = Mtd.shape[2]
    X = np.empty((n, len(keys) + 1 + nm), np.float32)
    for j, k_ in enumerate(keys):
        v = np.asarray(F[k_], np.float32)
        X[:, j] = v.reshape(-1) if v.ndim == 3 else np.repeat(v.reshape(-1), 6)
    X[:, len(keys)] = np.tile(np.arange(1, 7, dtype=np.float32), w * L)
    X[:, len(keys) + 1:] = np.repeat(Mtd.reshape(w * L, nm), 6, axis=0)
    NAMES = keys + ["h"] + [f"e{i}" for i in range(nm)]
    return X
def fmt_mtd(m, x, u): return np.column_stack([m, x, u]).astype(np.float32)

SPAN = 12 * 288; STEP = 1
TRX, TRY, TRK, TRN, TRS = [], [], [], 0, []
log("building train rows, mode", MODE)
FEB = 245
for P in PAN:
    L = P.L; d = P.S["train"]; V, Q, E, M = d["V"], d["Q"], d["E"], d["M"]; n = d["n"]
    Vmk = np.where(M, V, np.nan); Qmk = np.where(M, Q, np.nan)
    dn = np.r_[np.arange(1, L), -1]; up = np.r_[-1, np.arange(0, L - 1)]
    on_ok, og_c, pers_imp, pers_kn = candidates(P); og_ok = og_c & (pers_imp <= .9)
    P.TE_F = []
    nw = 0
    for sd in range(0, SPL["train"][1] - 8, STEP):
        if MODE == "feb" and not (sd < 200 - 12 or sd >= FEB): continue
        s0 = sd * 288; H = min(SPAN, n - s0)
        sel_on, sel_og = greedy(on_ok, og_ok, s0, s0 + H)
        if not sel_og: continue
        Vs = Vmk[s0:s0 + H].copy(); Qs = Qmk[s0:s0 + H].copy()
        for T in sel_on + sel_og: Vs[T - s0 + 1:T - s0 + 19] = np.nan; Qs[T - s0 + 1:T - s0 + 19] = np.nan
        for T in sel_on + sel_og: Vs[T - s0 - 12:T - s0] = V[T - 12:T]; Qs[T - s0 - 12:T - s0] = Q[T - 12:T]
        vw = View(Vs, Qs, E[s0:s0 + H], P.vcut, P.cap, dn, up)
        Ms = np.stack([fmt_mtd(*vw.feats(T - s0, 0)) for T in sel_og], 0).transpose(0, 2, 1) if False else None
        Mt = np.stack([fmt_mtd(*vw.feats(T - s0, 0)) for T in sel_og], 0)     # w x L x n
        X = feats_og(P, "train", np.array(sel_og), Mt)
        rr = np.array(sel_og)[:, None] + np.arange(1, 7)[None, :]
        y = P.lab[rr].reshape(-1); kn = P.known[rr].reshape(-1)
        # rows are (window, link, step): feats_og repeats per link then step; labels are (window, step, link), reorder
        y = P.lab[rr].transpose(0, 2, 1).reshape(-1); kn = P.known[rr].transpose(0, 2, 1).reshape(-1)
        if MODE == "feb" and sd >= FEB:
            P.TE_F.append((np.array(sel_og), X, Mt.shape[2]))
        else:
            TRX.append(X[kn]); TRY.append(y[kn].astype(np.float32)); TRN += int(kn.sum()); TRS.append(np.full(int(kn.sum()), sd % 3 == 0))
        nw += len(sel_og)
    log(P.name, "windows", nw, "rows so far", TRN)
    del Vmk, Qmk
Xall = np.concatenate(TRX); Yall = np.concatenate(TRY); Sall = np.concatenate(TRS); del TRX, TRY, TRS; gc.collect()
log("train rows", Xall.shape, "positives", float(Yall.mean()), "features", len(NAMES))
PARAMS = {"objective": "binary", "metric": "binary_logloss", "learning_rate": .05, "num_leaves": 63, "min_data_in_leaf": 60, "lambda_l2": 5, "feature_fraction": .7, "bagging_fraction": .7, "bagging_freq": 1, "num_threads": 4, "verbosity": -1, "seed": 20261105, "max_bin": 127}
NMA = np.array(NAMES); C_E = np.array([not (n_[0] == "e" and n_[1:].isdigit()) for n_ in NAMES]); C_X = np.array([not n_.startswith("x2_") for n_ in NAMES]); ROUNDS = 700
def fit(rows, cols, rounds=ROUNDS, **pr):
    b = lgb.train(dict(PARAMS, **pr), lgb.Dataset(Xall[rows][:, cols] if rows is not None else Xall[:, cols], label=Yall[rows] if rows is not None else Yall, feature_name=list(NMA[cols])), num_boost_round=rounds)
    return b
def iou_(p_, y, k): 
    a = p_ & k; b = y & k; u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u
THR = (.3, .35, .4, .45, .5, .55, .6)
def dec_(p, t_):
    s = p >= t_
    if not s.any(): s.flat[int(np.argmax(p))] = True
    return s
def table(name, rows):
    D = pd.DataFrame(rows); out = {"name": name, "n": len(D)}
    for c in D.columns:
        if c != "panel": out[c] = round(float(D.groupby("panel")[c].mean().mean()), 4)
    log(out); return out
RES = []
if MODE == "feb":
    ALLC = np.ones(len(NAMES), bool)
    EXPS = [("V1 every 3rd start, no x2, no evidence", Sall, C_E & C_X, {}, ROUNDS), ("V2 all starts, no x2, no evidence", None, C_E & C_X, {}, ROUNDS), ("V3 all starts, x2, no evidence", None, C_E, {}, ROUNDS),
            ("V4 all starts, x2, evidence", None, ALLC, {}, ROUNDS), ("V5 all starts, x2, no evidence, large", None, C_E, dict(learning_rate=.03, num_leaves=127, min_data_in_leaf=100), 1400)]
    PM = {}
    for nm_, rows, cols, pr, rd in EXPS:
        b = fit(rows, cols, rd, **pr); log("fitted", nm_)
        Rr = []
        for P in PAN:
            for Ts, X, _ in P.TE_F:
                p = b.predict(X[:, cols], num_threads=4).reshape(len(Ts), P.L, 6).transpose(0, 2, 1)
                for i, T in enumerate(Ts):
                    T = int(T); rr = T + np.arange(1, 7); y = P.lab[rr]; kn = P.known[rr]; PM.setdefault(nm_, []).append(p[i])
                    Rr.append(dict(panel=P.name, **{f"t{int(t_ * 100)}": iou_(dec_(p[i], t_), y, kn) for t_ in THR}))
        RES.append(table("Feb exact: " + nm_, Rr))
        if nm_.startswith(("V3", "V5")):
            imp = pd.Series(b.feature_importance("gain"), index=NMA[cols]).sort_values(ascending=False); log("top features " + " ".join(f"{a}:{v / imp.sum():.3f}" for a, v in imp.head(40).items()))
        del b; gc.collect()
    Rr = []; Ra = []; ii = 0
    for P in PAN:
        Pb = PubOng(P.name, PUBN.index(P.name)); idx = np.array([P.links.index(l) for l in Pb.links]); inv = np.argsort(idx)
        for Ts, X, _ in P.TE_F:
            for T in Ts:
                T = int(T); rr = T + np.arange(1, 7); y = P.lab[rr]; kn = P.known[rr]; pp = pub_pred(P, Pb, idx, inv, "train", T); pa = .5 * PM[EXPS[2][0]][ii] + .5 * PM[EXPS[4][0]][ii]; ii += 1
                Rr.append(dict(panel=P.name, **{f"t{int(t_ * 100)}": iou_(dec_(pp, t_), y, kn) for t_ in THR}))
                Ra.append(dict(panel=P.name, **{f"t{int(t_ * 100)}": iou_(dec_(pa, t_), y, kn) for t_ in THR}))
    RES += [table("Feb exact: public", Rr), table("Feb exact: mean of V3 and V5", Ra)]
else:
    # March pseudo-months: the selector on visible cells, histories through the masked layer, outside scored windows
    C_A = np.ones(len(NAMES), bool); LARGE = dict(learning_rate=.03, num_leaves=127, min_data_in_leaf=100)
    b1 = fit(None, C_A, 1400, **LARGE); log("fitted evidence"); b1.save_model(_os.path.join(TFB_OUT, "og_gbm_evidence.txt"))
    b0 = fit(None, C_E, 1400, **LARGE); log("fitted no evidence"); b0.save_model(_os.path.join(TFB_OUT, "og_gbm_noevidence.txt"))
    imp = pd.Series(b1.feature_importance("gain"), index=NMA[C_A]).sort_values(ascending=False); log("top features " + " ".join(f"{a}:{v / imp.sum():.3f}" for a, v in imp.head(30).items()))
    R1, RP, RB, RPER = [], [], [], []; R0 = []; RM = []
    for P in PAN:
        L = P.L; d = P.S["validation"]; V, Q, E = d["V"], d["Q"], d["E"]; nm_ = d["n"]
        Pb = PubOng(P.name, PUBN.index(P.name)); idx = np.array([P.links.index(l) for l in Pb.links]); inv = np.argsort(idx)
        dn = np.r_[np.arange(1, L), -1]; up = np.r_[-1, np.arange(0, L - 1)]
        bad = np.zeros(nm_, bool)
        for t_ in d["win"].t_idx.to_numpy(): bad[max(t_ - 12, 0):min(t_ + 20, nm_)] = True
        V2 = V.copy(); Q2 = Q.copy(); V2[bad] = np.nan; Q2[bad] = np.nan
        kn = np.isfinite(V2) & E; q = (np.nan_to_num(V2, nan=1e9) <= P.vcut[None, :]) & kn; a = q.any(1)
        on_m = np.zeros(nm_, bool); og_m = np.zeros(nm_, bool); run = 0
        for t0 in range(nm_):
            if a[t0]:
                if run >= 18 and t0 >= 18 and not bad[t0 - 18:t0 + 1].any(): on_m[t0 - 6] = True
                run = 0
            else:
                run += 1
        for T in range(12, nm_ - 7):
            if bad[T - 12:T + 7].any() or not a[T] or not a[T - 1] or not a[T + 1:T + 7].any() or np.isfinite(V2[T - 12:T]).mean() < .4: continue
            q0 = q[T]; qh = q[T + 1:T + 7]; kh = kn[T + 1:T + 7]; u_ = ((q0[None, :] & kh) | (qh & kh)).sum()
            if u_ and ((q0[None, :] & qh & kh).sum() / u_) <= .9: og_m[T] = True
        seen = set()
        for sd in range(9, 27):
            s0 = sd * 288; H = min(SPAN, nm_ - s0)
            sel_on, sel_og = greedy(on_m, og_m, s0, s0 + H)
            if not sel_og: continue
            Vs = V2[s0:s0 + H].copy(); Qs = Q2[s0:s0 + H].copy()
            for T in sel_on + sel_og: Vs[T - s0 + 1:T - s0 + 19] = np.nan; Qs[T - s0 + 1:T - s0 + 19] = np.nan
            vw = View(Vs, Qs, E[s0:s0 + H], P.vcut, P.cap, dn, up)
            Mt = np.stack([fmt_mtd(*vw.feats(T - s0, 0)) for T in sel_og], 0)
            X = feats_og(P, "validation", np.array(sel_og), Mt)
            p1 = b1.predict(X[:, C_A], num_threads=4).reshape(len(sel_og), L, 6).transpose(0, 2, 1); p0 = b0.predict(X[:, C_E], num_threads=4).reshape(len(sel_og), L, 6).transpose(0, 2, 1)
            for i, T in enumerate(sel_og):
                rr = T + np.arange(1, 7); y = q[rr]; kk = kn[rr]; pp = pub_pred(P, Pb, idx, inv, "validation", T)
                R1.append(dict(panel=P.name, **{f"t{int(t_ * 100)}": iou_(dec_(p1[i], t_), y, kk) for t_ in THR}))
                R0.append(dict(panel=P.name, **{f"t{int(t_ * 100)}": iou_(dec_(p0[i], t_), y, kk) for t_ in THR}))
                RM.append(dict(panel=P.name, **{f"t{int(t_ * 100)}": iou_(dec_(.5 * p0[i] + .5 * p1[i], t_), y, kk) for t_ in THR}))
                RP.append(dict(panel=P.name, **{f"t{int(t_ * 100)}": iou_(dec_(pp, t_), y, kk) for t_ in THR}))
                RB.append(dict(panel=P.name, **{f"t{int(t_ * 100)}": iou_(dec_(.5 * pp + .5 * p1[i], t_), y, kk) for t_ in THR}))
                RPER.append(dict(panel=P.name, per=iou_(np.tile(q[T], (6, 1)), y, kk), n_true=int((y & kk).sum())))
        log(P.name, "March pseudo windows", len(R1))
    RES += [table("March pseudo: with evidence", R1), table("March pseudo: no evidence", R0), table("March pseudo: mean of both", RM), table("March pseudo: public", RP), table("March pseudo: .5 public + .5 model", RB), table("March pseudo: persistence at T", RPER)]
    # scored rows
    key = pd.read_csv(f"{DATA}/submission_key.csv", dtype=str, usecols=["submission_id", "task", "window_id", "timestamp", "link_id"]); key = key[key.task == "queue"].drop(columns="task")
    out = []; CHECK = []; ONLINE_CHECK = os.environ.get("TFB_ONLINE_CHECK", "1") == "1"
    for P in PAN:
        L = P.L; dn = np.r_[np.arange(1, L), -1]; up = np.r_[-1, np.arange(0, L - 1)]
        for s_ in ("validation", "private"):
            d = P.S[s_]; vw = View(d["V"], d["Q"], d["E"], P.vcut, P.cap, dn, up)
            wn = d["win"]; wn = wn[wn.condition == "queue_ongoing"].sort_values("t_idx")
            Ts = wn.t_idx.to_numpy(); Mt = np.stack([fmt_mtd(*vw.feats(int(T), 0)) for T in Ts], 0)
            X = feats_og(P, s_, Ts, Mt); p1 = b1.predict(X[:, C_A], num_threads=4).reshape(len(Ts), L, 6).transpose(0, 2, 1); p0 = b0.predict(X[:, C_E], num_threads=4).reshape(len(Ts), L, 6).transpose(0, 2, 1)
            if ONLINE_CHECK:     # the organizers' check: remove every row stamped after T, recompute, compare
                for i, T in enumerate(Ts):
                    T = int(T); dc = dict(d)
                    for c_ in ("V", "Q", "O", "R"):
                        a_ = d[c_].copy(); a_[T + 1:] = np.nan; dc[c_] = a_
                    e_ = d["E"].copy(); e_[T + 1:] = False; dc["E"] = e_; P.S["_check"] = dc
                    Mc = fmt_mtd(*View(dc["V"], dc["Q"], dc["E"], P.vcut, P.cap, dn, up).feats(T, 0))[None]
                    Xc = feats_og(P, "_check", np.array([T]), Mc); n6 = L * 6; Xf = X[i * n6:(i + 1) * n6]
                    diff = float(np.nanmax(np.abs(Xc.astype(np.float64) - Xf.astype(np.float64)), initial=0.0)); nanpat = bool(np.array_equal(np.isnan(Xc), np.isnan(Xf)))
                    dp = float(np.max(np.abs(.5 * b1.predict(Xc[:, C_A], num_threads=4) + .5 * b0.predict(Xc[:, C_E], num_threads=4) - (.5 * p1[i] + .5 * p0[i]).T.reshape(-1))))
                    CHECK.append(dict(split=s_, T=T, max_abs_diff=diff, max_prob_diff=dp, same=bool(nanpat and np.allclose(Xc, Xf, rtol=1e-5, atol=1e-6, equal_nan=True))))     # tolerance: float sums differ in the last bits with memory alignment
                P.S.pop("_check", None)
            for i, w_ in enumerate(wn.itertuples(index=False)):
                To = pd.Timestamp(w_.forecast_origin); stamps = [(To + pd.Timedelta(minutes=5 * h)).strftime("%Y-%m-%dT%H:%M:%SZ") for h in range(1, 7)]
                out.append(pd.DataFrame({"window_id": w_.window_id, "timestamp": np.repeat(stamps, L), "link_id": np.tile(P.links, 6), "p_e": p1[i].reshape(-1), "p_n": p0[i].reshape(-1)}))
    if ONLINE_CHECK:
        ck = pd.DataFrame(CHECK); ck.to_csv(_os.path.join(TFB_OUT, "online_rule_check_ongoing.csv"), index=False)
        log("online-rule check (ongoing): windows", len(ck), "features unchanged with rows after T removed", int(ck.same.sum()), "max abs difference", float(ck.max_abs_diff.max()), "max probability difference", float(ck.max_prob_diff.max())); assert ck.same.all(), ck[~ck.same]
    pr = pd.concat(out, ignore_index=True); mm = key.merge(pr, on=["window_id", "timestamp", "link_id"], how="inner"); assert len(mm) == len(pr), (len(mm), len(pr))
    mm.to_csv(_os.path.join(TFB_OUT, "ongoing_probs.csv"), index=False); log("written", len(mm), "rows; positives at .5", int((mm.p_e >= .5).sum()), int((mm.p_n >= .5).sum()))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
log("results\n" + pd.DataFrame(RES).to_string(index=False))
log("done")
