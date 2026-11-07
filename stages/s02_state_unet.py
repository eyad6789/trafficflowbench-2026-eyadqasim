"""Stage 02 - Task 1 state U-Net (one seed per run, U_SEED). See the README for the method; Task 1 is offline, so month-own profiles use the whole month."""
# ---- paths: Kaggle by default; run_all.py points these at local folders -------------------------------------
import os as _os, glob as _glob
TFB_ROOTS = [r for r in _os.environ.get("TFB_INPUT", "/kaggle/input").split(_os.pathsep) if r]
TFB_OUT = _os.environ.get("TFB_OUT", "/kaggle/working"); _os.makedirs(TFB_OUT, exist_ok=True)
def G(pat):
    """every file matching pat under the input roots (recursive), sorted"""
    return sorted(set(p for r in TFB_ROOTS for p in _glob.glob(_os.path.join(r, "**", pat), recursive=True)))
# ----------------------------------------------------------------------------------------------------------------
import os
os.environ.setdefault('U_SEED', '2026')
# =====================================================================================================
# TrafficFlowBench 2026 - Task 1 (state reconstruction) + Task 3 (physics) : our own pipeline (eyadqasim)
#
#   1. Station x time grids from the released masked layer (anchors = released eligible cells).
#      In this release every mainline link carries exactly one detector station.
#   2. Leave-one-day-out weekday x time-of-day profiles from train (idea credit: yeashusemwal notebook).
#   3. Profile + interpolated-deviation bases; XGBoost (GPU) residual models for speed, flow/lane and
#      density/lane (L1, N-weighted), trained on the official train masks, early-stopped on late January,
#      evaluated on held-out February 2031 with the official S_state formula.
#   4. Physics step: fuse the dedicated density estimate into (v, q) and split the log-density
#      correction between flow and speed; tuned on the exact local objective
#      0.35*S_state + 0.15*(S_FD + 2*S_LWR)/3 (S_LWR identity proxy on observable links with mode-A ramp
#      validity, replicated from the official score_task3.py, MIT).
#      Density post-processing idea credit: honghanhhh notebook.
#   5. Inference on validation + private -> state_rows.csv (submission_id, speed_kmh, flow_vph) + metrics.json.
# =====================================================================================================
import os, sys, glob, json, time, gc, math, traceback, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)

CFG = dict(
    seed=2026,
    holdout_from="2031-02-01",          # held-out train days = local test (official masks)
    es_from="2031-01-18",               # early-stopping days (neither fitted nor evaluated)
    fit_frac=0.30,                      # share of train-template target cells used for fitting
    fit_cap_per_panel=400_000,
    es_cap_per_panel=40_000,
    chunk_days=30,
    decay=12.0,                         # one-sided deviation decay (steps)
    xgb=dict(max_depth=9, eta=0.06, subsample=0.8, colsample_bytree=0.7, min_child_weight=30,
             reg_lambda=3.0, max_bin=256),
    rounds=3000, early_stop=150,
    w_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
    a_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
)
np.random.seed(CFG["seed"])
OUT = TFB_OUT
cands = G("kaggle_public")
R = cands[0]
log("data root", R)
PANELS = [p["corridor_id"] for p in json.load(open(f"{R}/config/corridors.json"))["panels"]]
SPLITS = {"train": ("2030-06-01", "2031-03-01"), "validation": ("2031-03-01", "2031-04-01"),
          "private": ("2031-04-01", "2031-05-01")}
RATE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}
REGS = ("R1", "R2", "R3")

import xgboost as xgb
log("xgboost", xgb.__version__, "numpy", np.__version__, "pandas", pd.__version__)


# ----------------------------------------------------------------------------------- network helpers
def network_parameters(nd):
    """Replica of score_task3.network_parameters (official, MIT)."""
    n = pd.read_csv(f"{nd}/links.csv"); n.link_id = n.link_id.astype(str); n = n.drop_duplicates("link_id")
    out = n[["link_id", "length_km", "free_speed_kmh", "capacity_vph", "lanes"]].copy()
    for c, default in (("length_km", 1.0), ("free_speed_kmh", 100.0)):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(default)
    out["lanes"] = pd.to_numeric(out["lanes"], errors="coerce").fillna(1.0).clip(lower=1.0)
    out.capacity_vph = pd.to_numeric(out.capacity_vph, errors="coerce")
    lanes = pd.to_numeric(n.get("lanes", pd.Series(4.0, index=n.index)), errors="coerce").fillna(4.0)
    lane_fallback = pd.Series(1800.0 * lanes.to_numpy(), index=n.index)
    lane_fallback.index = out.index
    out.capacity_vph = out.capacity_vph.fillna(lane_fallback)
    f = pd.read_csv(f"{nd}/fd_parameters.csv"); f.link_id = f.link_id.astype(str)
    have = [c for c in ("free_speed_kmh", "capacity_vph", "lanes", "k_jam") if c in f.columns]
    m = f.groupby("link_id", as_index=False)[have].mean()
    out = out.merge(m, on="link_id", how="left", suffixes=("", "_fd"))
    for c in ("free_speed_kmh", "capacity_vph", "lanes"):
        if f"{c}_fd" in out.columns:
            out[c] = out[f"{c}_fd"].fillna(out[c]); out = out.drop(columns=[f"{c}_fd"])
    if "k_jam" not in out.columns:
        out["k_jam"] = np.nan
    out["critical_density"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
    out["k_jam"] = out.k_jam.fillna(100.0 * out.lanes.clip(lower=1.0))
    out["k_jam"] = np.maximum(out.k_jam, out.critical_density * 1.05)
    return out


def station_lanes(nd):
    """Replica of baseline_task1_historical_mean.station_lanes (scoring lanes)."""
    lanes_by_key = {}
    def absorb(frame, key):
        if key not in frame.columns or "lanes" not in frame.columns:
            return
        lanes = pd.to_numeric(frame["lanes"], errors="coerce")
        valid = lanes.notna() & (lanes >= 1) & (lanes <= 16)
        lanes_by_key.update(zip(frame[key][valid].astype(str), lanes[valid].astype(float)))
    absorb(pd.read_csv(f"{nd}/links.csv", dtype={"link_id": str}), "link_id")
    fr = pd.read_csv(f"{nd}/fd_parameters.csv", dtype={"station_id": str, "link_id": str})
    absorb(fr, "link_id"); absorb(fr, "station_id")
    return lanes_by_key


def split_ids(x):
    return {v.strip() for v in str(x).split(";") if v.strip()}


def ts_index(split):
    a, b = SPLITS[split]
    return pd.date_range(a, b, freq="5min", inclusive="left", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def files(P, split, kind):
    return sorted(glob.glob(f"{R}/corridors/{P}/{split}/{kind}/**/*.parquet", recursive=True))


class Panel:
    pass


# ----------------------------------------------------------------------------------- panel loading
def load_panel(pi, P):
    """Fill station x time grids file by file (one parquet = one day) to keep memory bounded."""
    t0 = time.time()
    nd = f"{R}/corridors/{P}/network"
    pn = Panel(); pn.P = P; pn.pi = pi
    topo = pd.read_csv(f"{nd}/lwr_mainline_topology.csv", dtype=str); topo["link_id"] = topo.link_id.astype(str)
    ordmap = dict(zip(topo.link_id, pd.to_numeric(topo.order_index, errors="coerce")))
    first = pd.read_parquet(files(P, "validation", "mainline_states_masked")[0], columns=["station_id", "link_id", "milepost"])
    st = first.astype({"station_id": str, "link_id": str}).drop_duplicates("station_id").copy()
    st["ord"] = st.link_id.map(ordmap).astype(float)
    st["milepost"] = pd.to_numeric(st.milepost, errors="coerce")
    st = st.sort_values(["ord", "milepost", "station_id"], na_position="last").reset_index(drop=True)
    pn.st = st.station_id.to_numpy(); pn.st_link = st.link_id.to_numpy()
    S = len(pn.st); pn.S = S
    sidx = pd.Index(pn.st)
    pn.links = pn.st_link.copy()                     # one station per link (asserted)
    assert len(set(pn.links)) == S, f"{P}: several stations share a link"
    lidx = pd.Index(pn.links); pn.L = S
    pn.st_li = np.arange(S)
    par = network_parameters(nd).set_index("link_id")
    lb = station_lanes(nd)
    by_st = pd.Series(pn.st).map(lb); by_li = pd.Series(pn.st_link).map(lb)
    pn.lanes = pd.to_numeric(by_st.fillna(by_li), errors="coerce").fillna(1.0).clip(lower=1.0).to_numpy(float)
    lp = par.reindex(pn.links)
    pn.l_has = lp[["length_km", "free_speed_kmh", "capacity_vph", "critical_density", "k_jam"]].notna().all(axis=1).to_numpy()
    pn.l_len = lp.length_km.to_numpy(float); pn.l_lanes = lp.lanes.to_numpy(float)
    pn.l_vf = lp.free_speed_kmh.to_numpy(float); pn.l_cap = lp.capacity_vph.to_numpy(float)
    pn.l_kc = lp.critical_density.to_numpy(float); pn.l_kj = lp.k_jam.to_numpy(float)
    pn.s_len = np.nan_to_num(pn.l_len, nan=1.0)
    pn.s_vf = np.nan_to_num(pn.l_vf, nan=100.0)
    pn.s_capl = np.nan_to_num(pn.l_cap / np.maximum(pn.l_lanes, 1), nan=2000.0)
    pn.s_kcl = np.nan_to_num(pn.l_kc / np.maximum(pn.l_lanes, 1), nan=20.0)
    pn.s_kjl = np.nan_to_num(pn.l_kj / np.maximum(pn.l_lanes, 1), nan=100.0)
    topo_ids = set(topo.link_id); state_links = set(pn.links[pn.l_has])
    obs = {}
    for r in topo.itertuples(index=False):
        obs[str(r.link_id)] = ((split_ids(getattr(r, "incoming_link_ids", "")) & topo_ids) <= state_links and
                               (split_ids(getattr(r, "outgoing_link_ids", "")) & topo_ids) <= state_links)
    pn.l_obs = np.array([bool(obs.get(l, False)) for l in pn.links])
    on_att = set(topo.loc[topo.on_ramp_link_ids.fillna("").astype(str).str.len() > 0, "link_id"])
    off_att = set(topo.loc[topo.off_ramp_link_ids.fillna("").astype(str).str.len() > 0, "link_id"])
    pn.l_on_att = np.array([l in on_att for l in pn.links]); pn.l_off_att = np.array([l in off_att for l in pn.links])
    pn.nb = np.full((S, 4), -1, np.int32)               # u1, u2, d1, d2 along the chain
    for c in range(S):
        for j, o in enumerate((-1, -2, 1, 2)):
            if 0 <= c + o < S:
                pn.nb[c, j] = c + o
    amap = pd.read_csv(f"{nd}/ramp_attachment_map.csv", dtype=str).rename(columns={"nearest_mainline_link_id": "link_id"})
    amap = amap[["ramp_link_id", "link_id"]].astype(str)
    MC = ["date", "timestamp", "station_id", "speed_kmh", "flow_vph", "occupancy",
          "density_occ_linear_vehpkm", "pct_observed", "is_score_eligible", "mask_regime"]
    pn.g = {}
    for split in SPLITS:
        times = ts_index(split); T = len(times)
        tix = pd.Index(times)
        g = {name: np.full((T, S), np.nan, np.float32) for name in ("v", "q", "o", "dn", "pct")}
        E = np.zeros((T, S), bool); dreg = {}
        for f in files(P, split, "mainline_states_masked"):
            d = pd.read_parquet(f, columns=MC)
            r_ = tix.get_indexer(d.timestamp.astype(str)); c_ = sidx.get_indexer(d.station_id.astype(str))
            ok = (r_ >= 0) & (c_ >= 0)
            assert ok.all(), f"{P} {split} {f}: {int((~ok).sum())} rows off-grid"
            for name, col in (("v", "speed_kmh"), ("q", "flow_vph"), ("o", "occupancy"),
                              ("dn", "density_occ_linear_vehpkm"), ("pct", "pct_observed")):
                g[name][r_, c_] = pd.to_numeric(d[col], errors="coerce").to_numpy(np.float32)
            E[r_, c_] = d.is_score_eligible.astype(bool).to_numpy()
            for dt_, rg in zip(d.date.astype(str).unique(), [str(d.mask_regime.iloc[0])] * d.date.nunique()):
                dreg[dt_] = rg
        g["elig"] = E
        g["reg"] = pd.Series(tix.str[:10]).map(dreg).fillna("NA").to_numpy()
        g["rate_row"] = pd.Series(g["reg"]).map(RATE).fillna(0.33).to_numpy(np.float32)
        g["dow_row"] = pd.to_datetime(tix, utc=True).weekday.to_numpy().astype(np.float32)
        g["times"] = np.asarray(times)
        tp = pd.read_csv(f"{R}/task1/{P}/{split}/sample_submission_state.csv", dtype=str,
                         usecols=["timestamp", "station_id", "link_id", "mask_regime"])
        tr = tix.get_indexer(tp.timestamp); tc = sidx.get_indexer(tp.station_id)
        assert (tr >= 0).all() and (tc >= 0).all(), f"{P} {split} template off-grid"
        assert (tp.link_id.to_numpy() == pn.st_link[tc]).all(), f"{P} {split} template link mismatch"
        assert (tp.mask_regime.to_numpy() == g["reg"][tr]).all(), f"{P} {split} template regime mismatch"
        g["tr"] = tr.astype(np.int32); g["tc"] = tc.astype(np.int32)
        TG = np.zeros((T, S), bool); TG[tr, tc] = True; g["TG"] = TG
        del tp
        on_f = np.zeros((T, S), np.float32); off_f = np.zeros((T, S), np.float32)
        on_n = np.zeros((T, S), np.int16); off_n = np.zeros((T, S), np.int16)
        for f in files(P, split, "ramp_states"):
            rd = pd.read_parquet(f, columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph", "pct_observed", "is_score_eligible"])
            rd["ramp_link_id"] = rd.ramp_link_id.astype(str)
            rd["flow_vph"] = pd.to_numeric(rd.flow_vph, errors="coerce"); rd["pct_observed"] = pd.to_numeric(rd.pct_observed, errors="coerce")
            rd = rd[rd.is_score_eligible.astype(bool) & (rd.pct_observed >= 75) & rd.flow_vph.notna()]
            rd = rd.merge(amap, on="ramp_link_id", how="inner")
            typ = rd.ramp_type.astype(str).str.upper()
            io = typ.isin(["OR", "ON"]).to_numpy(); ifo = typ.isin(["FR", "OFF"]).to_numpy()
            rr = tix.get_indexer(rd.timestamp.astype(str)); rl = lidx.get_indexer(rd.link_id)
            okr = (rr >= 0) & (rl >= 0)
            fv = rd.flow_vph.to_numpy(np.float32)
            a1 = okr & io; a2 = okr & ifo
            np.add.at(on_f, (rr[a1], rl[a1]), fv[a1]); np.add.at(on_n, (rr[a1], rl[a1]), 1)
            np.add.at(off_f, (rr[a2], rl[a2]), fv[a2]); np.add.at(off_n, (rr[a2], rl[a2]), 1)
        g["on_f"], g["off_f"], g["on_n"], g["off_n"] = on_f, off_f, on_n, off_n
        g["ramp_ok"] = ((on_n > 0) | ~pn.l_on_att[None, :]) & ((off_n > 0) | ~pn.l_off_att[None, :])
        pn.g[split] = g
    g = pn.g["train"]; times = g["times"]; T = len(times); tix = pd.Index(times)
    Vt = np.full((T, S), np.nan, np.float32); Qt = Vt.copy(); Et = np.zeros((T, S), bool)
    for f in files(P, "train", "mainline_states"):
        d = pd.read_parquet(f, columns=["timestamp", "station_id", "speed_kmh", "flow_vph", "is_score_eligible"])
        r_ = tix.get_indexer(d.timestamp.astype(str)); c_ = sidx.get_indexer(d.station_id.astype(str))
        ok = (r_ >= 0) & (c_ >= 0)
        Vt[r_[ok], c_[ok]] = pd.to_numeric(d.speed_kmh, errors="coerce").to_numpy(np.float32)[ok]
        Qt[r_[ok], c_[ok]] = pd.to_numeric(d.flow_vph, errors="coerce").to_numpy(np.float32)[ok]
        Et[r_[ok], c_[ok]] = d.is_score_eligible.astype(bool).to_numpy()[ok]
    Et &= np.isfinite(Vt) & np.isfinite(Qt)
    g["Vt"], g["Qt"], g["Et"] = Vt, Qt, Et
    gc.collect()
    log(f"loaded {P}: S={S} obs_links={int(pn.l_obs.sum())} on/off attached={int(pn.l_on_att.sum())}/{int(pn.l_off_att.sum())} "
        f"targets train/val/priv={len(pn.g['train']['tr'])}/{len(pn.g['validation']['tr'])}/{len(pn.g['private']['tr'])} "
        f"({time.time() - t0:.0f}s)")
    return pn


# ----------------------------------------------------------------------------------- profiles
def slots_of(times):
    ts = pd.to_datetime(pd.Index(times), utc=True)
    return (ts.weekday.to_numpy() * 288 + ts.hour.to_numpy() * 12 + ts.minute.to_numpy() // 5).astype(np.int32)


def build_profiles(pn):
    """Weekday x time-of-day means (+-1 slot smoothing) from train truth; leave-one-day-out grid for train."""
    g = pn.g["train"]
    slot = slots_of(g["times"])
    Et = g["Et"]
    for name in ("v", "q", "k"):
        if name == "k":
            X = g["Qt"] / np.maximum(g["Vt"], 1.0)
        else:
            X = g["Vt"] if name == "v" else g["Qt"]
        x = np.where(Et, X, 0.0).astype(np.float64); e = Et.astype(np.float64)
        sums = np.zeros((2016, pn.S)); cnts = np.zeros((2016, pn.S))
        np.add.at(sums, slot, x); np.add.at(cnts, slot, e)
        s3 = sums + np.roll(sums, 1, 0) + np.roll(sums, -1, 0)
        c3 = cnts + np.roll(cnts, 1, 0) + np.roll(cnts, -1, 0)
        overall = sums.sum(0) / np.maximum(cnts.sum(0), 1)
        own = x.copy(); own[1:] += x[:-1]; own[:-1] += x[1:]
        owne = e.copy(); owne[1:] += e[:-1]; owne[:-1] += e[1:]
        num = s3[slot] - own; den = c3[slot] - owne
        g["P" + name] = np.where(den > 0.5, num / np.maximum(den, 1e-9), overall[None, :]).astype(np.float32)
        del x, e, own, owne, num, den
        for split in ("validation", "private"):
            gs = pn.g[split]; sl = slots_of(gs["times"])
            gs["P" + name] = np.where(c3[sl] > 0.5, s3[sl] / np.maximum(c3[sl], 1e-9), overall[None, :]).astype(np.float32)
        gc.collect()


# ----------------------------------------------------------------------------------- features
GRID_KEYS = ("v", "q", "o", "dn", "pct", "elig", "TG", "Pv", "Pq", "Pk", "on_f", "off_f", "on_n", "off_n")


def slice_grid(g, r0, r1):
    s = {k: g[k][r0:r1] for k in GRID_KEYS}
    for k in ("reg", "rate_row", "dow_row", "times"):
        s[k] = g[k][r0:r1]
    return s


def prev_next(A):
    T = A.shape[0]
    pos = np.arange(T, dtype=np.int32)[:, None]
    prev = np.maximum.accumulate(np.where(A, pos, np.int32(-1)), axis=0).astype(np.int32)
    nxt = np.minimum.accumulate(np.where(A, pos, np.int32(T))[::-1], axis=0)[::-1].astype(np.int32)
    return prev, nxt


def win_mean(D, A, w):
    T = D.shape[0]
    x = np.where(A, D, 0.0).astype(np.float64); c = A.astype(np.float64)
    z = np.zeros((1, D.shape[1]))
    cx = np.concatenate([z, np.cumsum(x, 0)]); cc = np.concatenate([z, np.cumsum(c, 0)])
    hi = np.minimum(np.arange(T) + w + 1, T); lo = np.maximum(np.arange(T) - w, 0)
    sx = cx[hi] - cx[lo]; sc = cc[hi] - cc[lo]
    return np.where(sc > 0, sx / np.maximum(sc, 1), np.nan).astype(np.float32), sc.astype(np.float32)


def grid_bases(g, pn):
    """Anchor structure + profile/deviation bases on a (chunk of a) split grid."""
    V, Q = g["v"], g["q"]
    fin = np.isfinite(V) & np.isfinite(Q)
    A = fin & g["elig"] & ~g["TG"]
    T, S = V.shape
    prev, nxt = prev_next(A)
    cols = np.arange(S)[None, :]
    t = np.arange(T, dtype=np.int32)[:, None]
    hp = prev >= 0; hn = nxt < T
    pp = np.where(hp, prev, 0); nn = np.where(hn, nxt, 0)
    g1 = np.where(hp, t - pp, 999).astype(np.float32); g2 = np.where(hn, nn - t, 999).astype(np.float32)
    lanes = pn.lanes[None, :].astype(np.float32)
    K = np.where(A, Q / np.maximum(V, 1.0), np.nan).astype(np.float32)
    b = {"A": A, "AQ": fin & ~g["elig"], "prev": prev, "nxt": nxt, "g1": g1, "g2": g2, "hp": hp, "hn": hn, "pp": pp, "nn": nn}
    both = hp & hn
    wgt = np.where(both, g1 / np.maximum(g1 + g2, 1), 0).astype(np.float32)
    dec = CFG["decay"]
    days = T // 288
    for name, X in (("v", V), ("q", Q), ("k", K)):
        P_ = g["P" + name]
        D = np.where(A, X - P_, np.nan).astype(np.float32)
        dp = D[pp, cols]; dn_ = D[nn, cols]
        dev = np.where(both, dp * (1 - wgt) + dn_ * wgt,
                       np.where(hp, dp * np.exp(-g1 / dec), np.where(hn, dn_ * np.exp(-g2 / dec), 0.0)))
        b["base" + name] = np.where(A, X, P_ + dev).astype(np.float32)
        xp = X[pp, cols]; xn = X[nn, cols]
        b["lin" + name] = np.where(both, xp * (1 - wgt) + xn * wgt, np.nan).astype(np.float32)
        b["D" + name] = D
        b["w6" + name], c6 = win_mean(D, A, 6)
        b["w24" + name], _ = win_mean(D, A, 24)
        Dl = D if name == "v" else D / lanes
        with np.errstate(all="ignore"):
            b["cor" + name] = np.nanmean(Dl, axis=1).astype(np.float32)
            b["day" + name] = np.repeat(np.nanmean(D.reshape(days, 288, S), axis=1), 288, axis=0).astype(np.float32)
        del dp, dn_, dev, xp, xn
    b["c6"] = c6
    b["corfrac"] = A.mean(axis=1).astype(np.float32)
    # nearest anchored station upstream / downstream at the same time step
    cpos = np.arange(S, dtype=np.int32)[None, :]
    lp = np.maximum.accumulate(np.where(A, cpos, np.int32(-1)), axis=1)           # last anchored col <= c
    rn = np.minimum.accumulate(np.where(A, cpos, np.int32(S))[:, ::-1], axis=1)[:, ::-1]   # first anchored col >= c
    b["su"] = np.concatenate([np.full((T, 1), -1, np.int32), lp[:, :-1]], axis=1)   # strictly upstream
    b["sd"] = np.concatenate([rn[:, 1:], np.full((T, 1), S, np.int32)], axis=1)     # strictly downstream
    return b


FEATS = []


def row_features(g, b, pn, rows, cols):
    """Feature matrix for target cells (rows, cols) of a (chunk) grid."""
    T, S = g["v"].shape
    V, Q, O, DN, PCT = g["v"], g["q"], g["o"], g["dn"], g["pct"]
    lan = pn.lanes.astype(np.float32)
    l_c = lan[cols]
    F = {}
    nanw = lambda m, x: np.where(m, x, np.nan).astype(np.float32)
    g1 = b["g1"][rows, cols]; g2 = b["g2"][rows, cols]
    pp = b["pp"][rows, cols]; nn = b["nn"][rows, cols]
    hp = b["hp"][rows, cols]; hn = b["hn"][rows, cols]
    F["g1"] = np.minimum(g1, 300); F["g2"] = np.minimum(g2, 300)
    F["gmin"] = np.minimum(F["g1"], F["g2"]); F["gsum"] = np.minimum(g1 + g2, 600)
    F["v_p1"] = nanw(hp, V[pp, cols]); F["v_n1"] = nanw(hn, V[nn, cols])
    F["q_p1"] = nanw(hp, Q[pp, cols] / l_c); F["q_n1"] = nanw(hn, Q[nn, cols] / l_c)
    F["o_p1"] = nanw(hp, O[pp, cols]); F["o_n1"] = nanw(hn, O[nn, cols])
    F["dn_p1"] = nanw(hp, DN[pp, cols] / l_c); F["dn_n1"] = nanw(hn, DN[nn, cols] / l_c)
    F["pct_p1"] = nanw(hp, PCT[pp, cols]); F["pct_n1"] = nanw(hn, PCT[nn, cols])
    F["pct_self"] = PCT[rows, cols].astype(np.float32)              # quality fields survive the mask
    for nm in ("v", "q", "k"):
        sc = 1.0 if nm == "v" else l_c
        F[f"d{nm}_p1"] = nanw(hp, b["D" + nm][pp, cols] / sc); F[f"d{nm}_n1"] = nanw(hn, b["D" + nm][nn, cols] / sc)
    prev, nxt = b["prev"], b["nxt"]
    p2 = np.where(hp & (pp > 0), prev[np.maximum(pp - 1, 0), cols], -1)
    n2 = np.where(hn & (nn < T - 1), nxt[np.minimum(nn + 1, T - 1), cols], T)
    h2p = p2 >= 0; h2n = n2 < T
    p2c = np.maximum(p2, 0); n2c = np.minimum(n2, T - 1)
    F["v_p2"] = nanw(h2p, V[p2c, cols]); F["v_n2"] = nanw(h2n, V[n2c, cols])
    F["q_p2"] = nanw(h2p, Q[p2c, cols] / l_c); F["q_n2"] = nanw(h2n, Q[n2c, cols] / l_c)
    F["g_p2"] = np.where(h2p, rows - p2, 999).astype(np.float32); F["g_n2"] = np.where(h2n, n2 - rows, 999).astype(np.float32)
    # ineligible-but-released values right next to the target
    AQ = b["AQ"]
    for off in (-1, 1):
        rr = np.clip(rows + off, 0, T - 1)
        m = AQ[rr, cols] & (rows + off >= 0) & (rows + off < T)
        F[f"aq_v{off:+d}"] = nanw(m, V[rr, cols]); F[f"aq_q{off:+d}"] = nanw(m, Q[rr, cols] / l_c)
        F[f"aq_pct{off:+d}"] = nanw(m, PCT[rr, cols])
    for nm in ("v", "q", "k"):
        sc = 1.0 if nm == "v" else l_c
        F["base_" + nm] = b["base" + nm][rows, cols] / sc
        F["lin_" + nm] = b["lin" + nm][rows, cols] / sc
        F["P_" + nm] = g["P" + nm][rows, cols] / sc
        F["w6_" + nm] = b["w6" + nm][rows, cols] / sc
        F["w24_" + nm] = b["w24" + nm][rows, cols] / sc
        F["day_" + nm] = b["day" + nm][rows, cols] / sc
        F["cor_" + nm] = b["cor" + nm][rows]
    F["c6"] = b["c6"][rows, cols]
    F["corfrac"] = b["corfrac"][rows]
    rp = np.minimum(rows + 1, T - 1); rm = np.maximum(rows - 1, 0)
    F["Pv_slope"] = g["Pv"][rp, cols] - g["Pv"][rm, cols]
    F["Pq_slope"] = (g["Pq"][rp, cols] - g["Pq"][rm, cols]) / l_c
    A = b["A"]
    for j, nm in enumerate(("u1", "u2", "d1", "d2")):
        nc = pn.nb[cols, j]; has = nc >= 0; ncc = np.maximum(nc, 0)
        ln = lan[ncc]
        F[nm + "_lr"] = nanw(has, ln / l_c)
        F[nm + "_len"] = nanw(has, pn.s_len[ncc])
        at = has & A[rows, ncc]
        F[nm + "_v"] = nanw(at, V[rows, ncc]); F[nm + "_q"] = nanw(at, Q[rows, ncc] / ln)
        F[nm + "_dn"] = nanw(at, DN[rows, ncc] / ln)
        F[nm + "_dv"] = nanw(at, b["Dv"][rows, ncc]); F[nm + "_dq"] = nanw(at, b["Dq"][rows, ncc] / ln)
        F[nm + "_bv"] = nanw(has, b["basev"][rows, ncc]); F[nm + "_bq"] = nanw(has, b["baseq"][rows, ncc] / ln)
        for off in (-1, 1):
            rr = np.clip(rows + off, 0, T - 1)
            a2 = has & A[rr, ncc]
            F[f"{nm}_v{off:+d}"] = nanw(a2, V[rr, ncc]); F[f"{nm}_q{off:+d}"] = nanw(a2, Q[rr, ncc] / ln)
    for nm, key in (("su", "su"), ("sd", "sd")):
        nc = b[key][rows, cols]; has = (nc >= 0) & (nc < S); ncc = np.clip(nc, 0, S - 1)
        ln = lan[ncc]
        F[nm + "_hops"] = np.where(has, np.abs(nc - cols), 99).astype(np.float32)
        F[nm + "_km"] = nanw(has, np.abs(np.cumsum(pn.s_len)[ncc] - np.cumsum(pn.s_len)[cols]))
        F[nm + "_v"] = nanw(has, V[rows, ncc]); F[nm + "_q"] = nanw(has, Q[rows, ncc] / ln)
        F[nm + "_dv"] = nanw(has, b["Dv"][rows, ncc]); F[nm + "_dq"] = nanw(has, b["Dq"][rows, ncc] / ln)
    F["on_f"] = g["on_f"][rows, cols] / l_c; F["off_f"] = g["off_f"][rows, cols] / l_c
    F["on_n"] = g["on_n"][rows, cols].astype(np.float32); F["off_n"] = g["off_n"][rows, cols].astype(np.float32)
    F["on_att"] = pn.l_on_att[cols].astype(np.float32); F["off_att"] = pn.l_off_att[cols].astype(np.float32)
    for j, nm in ((0, "u1"), (2, "d1")):
        nc = pn.nb[cols, j]; has = nc >= 0; ncc = np.maximum(nc, 0)
        F[nm + "_net"] = nanw(has, (g["on_f"][rows, ncc] - g["off_f"][rows, ncc]) / l_c)
    F["tod"] = (rows % 288).astype(np.float32)
    F["dow"] = g["dow_row"][rows]
    F["rate"] = g["rate_row"][rows]
    F["panel"] = np.full(len(rows), pn.pi, np.float32)
    F["spos"] = (cols / max(pn.S - 1, 1)).astype(np.float32)
    F["lanes"] = l_c; F["vf"] = pn.s_vf[cols].astype(np.float32); F["capl"] = pn.s_capl[cols].astype(np.float32)
    F["kcl"] = pn.s_kcl[cols].astype(np.float32); F["kjl"] = pn.s_kjl[cols].astype(np.float32)
    F["len"] = pn.s_len[cols].astype(np.float32)
    global FEATS
    if not FEATS:
        FEATS = list(F.keys())
    X = np.empty((len(rows), len(FEATS)), np.float32)
    for i, k in enumerate(FEATS):
        X[:, i] = F[k]
    X[~np.isfinite(X)] = np.nan
    return X


# ----------------------------------------------------------------------------------- scoring
def physics_setup(pn):
    """Held-out (February) truth grids for the exact S_FD / S_LWR proxy. Station == link here."""
    g = pn.g["train"]
    sel = np.where(g["times"] >= CFG["holdout_from"])[0]
    E = g["Et"][sel]
    ph = dict(off=int(sel[0]), E=E, TG=g["TG"][sel] & E, Vt=g["Vt"][sel], Qt=g["Qt"][sel], reg=g["reg"][sel],
              ramp_ok=g["ramp_ok"][sel])
    ph["valid"] = E & pn.l_has[None, :]
    ph["Nt"] = np.where(E, g["Qt"][sel] / np.maximum(g["Vt"][sel], 1.0), np.nan) * pn.l_len[None, :]
    return ph


def physics_score(pn, ph, Vp, Qp):
    """Exact S_FD and S_LWR identity proxy per regime (rhs = dN_true, i.e. organizer flux projected on truth)."""
    TG = ph["TG"]
    Vm = np.where(TG, Vp, ph["Vt"]).astype(np.float64); Qm = np.where(TG, Qp, ph["Qt"]).astype(np.float64)
    k = Qm / np.maximum(Vm, 1.0); N = k * pn.l_len[None, :]
    lanes = np.maximum(pn.l_lanes, 1.0)[None, :]
    ql = Qm / lanes; kl = k / lanes
    kc = (pn.l_kc / np.maximum(pn.l_lanes, 1))[None, :]; kj = (pn.l_kj / np.maximum(pn.l_lanes, 1))[None, :]
    cp = (pn.l_cap / np.maximum(pn.l_lanes, 1))[None, :]
    with np.errstate(all="ignore"):
        qfd = np.where(kl <= kc, pn.l_vf[None, :] * kl, cp / np.maximum(kj - kc, 1e-9) * np.maximum(kj - kl, 0))
    valid = ph["valid"]
    dN = N[1:] - N[:-1]; dNt = ph["Nt"][1:] - ph["Nt"][:-1]
    tr = valid[1:] & valid[:-1] & pn.l_obs[None, :] & ph["ramp_ok"][:-1]
    out = {}
    for r in REGS:
        rr = ph["reg"] == r
        if not rr.any():
            continue
        u = valid & rr[:, None]
        s_fd = max(0.0, 1.0 - np.abs(ql - qfd)[u].sum() / (np.abs(ql)[u].sum() + 1e-9))
        sub = u & TG
        if sub.any() and (Qm[sub] < 50).mean() > 0.2:
            s_fd = 0.0
        t2 = tr & (rr[1:] & rr[:-1])[:, None]
        num = np.abs(dN - dNt)[t2].sum(); den = np.abs(dNt)[t2].sum()
        e = num / (den + 1e-9)
        out[r] = dict(S_FD=s_fd, S_LWR=max(0.0, 1.0 - min(1.0, e)), E_LWR=e, n_trans=int(t2.sum()))
    return out


def state_score(pn, cols, regs, vp, qp, vt, qt):
    lanes = pn.lanes[cols]
    out = {}
    for r in REGS:
        m = regs == r
        if not m.any():
            continue
        rv = float(np.sqrt(np.mean((vp[m] - vt[m]) ** 2))); rq = float(np.sqrt(np.mean(((qp[m] - qt[m]) / lanes[m]) ** 2)))
        out[r] = dict(rmse_v=rv, rmse_q=rq, S_state=0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600), n=int(m.sum()))
    return out


def write_state_rows(panels, TM, vcol, qcol, fname):
    keys, sp, fl = [], [], []
    for pn in panels:
        for split in ("validation", "private"):
            gs = pn.g[split]; m = ((TM.pi == pn.pi) & (TM.split == split)).to_numpy()
            r_, c_ = TM.row.to_numpy()[m], TM.col.to_numpy()[m]
            assert (r_ == gs["tr"]).all() and (c_ == gs["tc"]).all()
            k = (pd.Series(gs["times"][r_]).radd(f"{split}|{pn.P}|") + "|" + pd.Series(pn.st[c_]) + "|" +
                 pd.Series(pn.st_link[c_]) + "|" + pd.Series(gs["reg"][r_]))
            keys.append(k); sp.append(TM[vcol].to_numpy()[m]); fl.append(TM[qcol].to_numpy()[m])
    sk = pd.Index(pd.concat(keys, ignore_index=True)); sp = np.concatenate(sp); fl = np.concatenate(fl)
    assert sk.is_unique, "duplicate state keys"
    ids, pos = [], []
    for ch in pd.read_csv(f"{R}/submission_key.csv", dtype=str, chunksize=1_000_000,
                          usecols=["submission_id", "task", "split", "panel", "timestamp", "station_id", "link_id", "mask_regime"]):
        ch = ch[ch.task == "state"]
        kk = ch.split + "|" + ch.panel + "|" + ch.timestamp + "|" + ch.station_id + "|" + ch.link_id + "|" + ch.mask_regime
        ids.append(ch.submission_id.astype(np.int64).to_numpy()); pos.append(sk.get_indexer(kk))
    ids = np.concatenate(ids); pos = np.concatenate(pos)
    n_miss = int((pos < 0).sum())
    log(f"state key rows {len(ids)}, our rows {len(sk)}, missing {n_miss}, unused {len(sk) - len(np.unique(pos[pos >= 0]))}")
    assert n_miss == 0, "missing state predictions"
    out = pd.DataFrame({"submission_id": ids, "speed_kmh": sp[pos], "flow_vph": fl[pos]}).sort_values("submission_id")
    assert np.isfinite(out.speed_kmh).all() and np.isfinite(out.flow_vph).all() and (out.flow_vph >= 0).all()
    out.to_csv(f"{OUT}/{fname}", index=False, float_format="%.4f")
    low = float((out.flow_vph < 50).mean())
    log(f"wrote {fname} {len(out)} rows; speed {out.speed_kmh.min():.1f}..{out.speed_kmh.max():.1f}; "
        f"flow {out.flow_vph.min():.0f}..{out.flow_vph.max():.0f}; low-flow share {low:.5f}")
    return len(out), low


PN_ARRAYS = ("st", "st_link", "lanes", "l_has", "l_len", "l_lanes", "l_vf", "l_cap", "l_kc", "l_kj", "s_len", "s_vf",
             "s_capl", "s_kcl", "s_kjl", "l_obs", "l_on_att", "l_off_att", "nb")


def save_panel_meta(pn, path):
    """Everything the blend / physics stage needs, so later kernels do not reload the raw data."""
    d = {k: getattr(pn, k) for k in PN_ARRAYS}
    for k, v in pn.ph.items():
        d["ph_" + k] = np.asarray(v)
    for split in ("validation", "private"):
        for k in ("times", "reg", "tr", "tc"):
            d[f"{split}_{k}"] = np.asarray(pn.g[split][k])
    d["P"] = np.asarray(pn.P); d["pi"] = np.asarray(pn.pi)
    np.savez_compressed(path, **d)


def load_panel_meta(path):
    z = np.load(path, allow_pickle=True)   # our own files (string arrays are object dtype)
    pn = Panel()
    for k in PN_ARRAYS:
        setattr(pn, k, z[k])
    pn.P = str(z["P"]); pn.pi = int(z["pi"]); pn.S = len(pn.st); pn.L = pn.S; pn.st_li = np.arange(pn.S)
    pn.links = pn.st_link
    pn.ph = {k[3:]: (int(z[k]) if k == "ph_off" else z[k]) for k in z.files if k.startswith("ph_")}
    pn.g = {split: {k: z[f"{split}_{k}"] for k in ("times", "reg", "tr", "tc")} for split in ("validation", "private")}
    return pn


# ----------------------------------------------------------------------------------- shared evaluation + physics step
def make_evaluator(panels, H):
    """Exact held-out evaluator: official S_state per panel/regime, S_FD + S_LWR identity proxy, combined objective."""
    Hp = {}
    for pn in panels:
        m = (H.pi == pn.pi).to_numpy()
        Hp[pn.pi] = dict(m=m, cols=H.col.to_numpy()[m], regs=H.reg.to_numpy()[m], vt=H.vt.to_numpy()[m],
                         qt=H.qt.to_numpy()[m], rr=H.row.to_numpy()[m] - pn.ph["off"])

    def evaluate(vh, qh, detail=False):
        rows = []
        for pn in panels:
            hp_ = Hp[pn.pi]; m = hp_["m"]; ph = pn.ph
            ss = state_score(pn, hp_["cols"], hp_["regs"], vh[m], qh[m], hp_["vt"], hp_["qt"])
            Vp = np.full(ph["Vt"].shape, np.nan, np.float64); Qp = Vp.copy()
            Vp[hp_["rr"], hp_["cols"]] = vh[m]; Qp[hp_["rr"], hp_["cols"]] = qh[m]
            ps = physics_score(pn, ph, Vp, Qp)
            for r in REGS:
                if r in ss and r in ps:
                    rows.append(dict(panel=pn.P, regime=r, **ss[r], **ps[r]))
        d = pd.DataFrame(rows)
        d["S_phys"] = (d.S_FD + 2 * d.S_LWR) / 3
        d["obj"] = 0.35 * d.S_state + 0.15 * d.S_phys
        pm = d.groupby("panel")[["S_state", "S_FD", "S_LWR", "S_phys"]].mean()
        res = dict(S_state=float(pm.S_state.mean()), S_FD=float(pm.S_FD.mean()), S_LWR=float(pm.S_LWR.mean()),
                   S_physics=float(pm.S_phys.mean()))
        res["objective"] = 0.35 * res["S_state"] + 0.15 * res["S_physics"]
        reg = d.groupby("regime")[["rmse_v", "rmse_q", "S_state", "S_FD", "S_LWR", "S_phys", "E_LWR", "obj"]].mean()
        res["per_regime"] = {r: {k: float(v) for k, v in reg.loc[r].items()} for r in reg.index}
        return (res, d) if detail else res
    return evaluate


def adjust(v, q, k, w, a):
    """Move log-density toward the dedicated density estimate by weight w; share a of the move goes to flow."""
    d = w * (np.log(np.maximum(k, 1e-3)) - np.log(np.maximum(q, 1.0) / np.maximum(v, 1.0)))
    d = np.clip(d, -1.0, 1.0)
    return np.clip(v * np.exp(-(1 - a) * d), 3, 140), np.clip(q * np.exp(a * d), 0, None)


def apply_physics(v, q, k, regs, iso, params):
    """params[(regime, 'iso'|'run')] = (w, a)."""
    vo, qo = v.copy(), q.copy()
    for (r, kind), (w, a) in params.items():
        m = (regs == r) & (iso if kind == "iso" else ~iso)
        if m.any() and w > 0:
            vo[m], qo[m] = adjust(v[m], q[m], k[m], w, a)
    return vo, qo


def tune_physics(evaluate, v, q, k, regs, iso, w_grid=(0, 0.25, 0.5, 0.75, 1.0), a_grid=(0.25, 0.5, 0.75, 1.0)):
    """Per-regime choice is exact (regimes are scored separately); iso/run weights by a joint grid per regime."""
    log_rows = []
    best = {r: (-1.0, (0.0, 0.0, 0.0)) for r in REGS}
    for a in a_grid:
        for wi in w_grid:
            for wr in w_grid:
                if (wi == 0 and wr == 0) and a != a_grid[0]:
                    continue
                params = {}
                for r in REGS:
                    params[(r, "iso")] = (wi, a); params[(r, "run")] = (wr, a)
                vv, qq = apply_physics(v, q, k, regs, iso, params)
                res = evaluate(vv, qq)
                for r in REGS:
                    if r in res["per_regime"]:
                        o = res["per_regime"][r]["obj"]
                        log_rows.append(dict(a=a, w_iso=wi, w_run=wr, regime=r, obj=o, S_state=res["per_regime"][r]["S_state"],
                                             S_LWR=res["per_regime"][r]["S_LWR"]))
                        if o > best[r][0]:
                            best[r] = (o, (wi, wr, a))
    params = {}
    for r, (o, (wi, wr, a)) in best.items():
        params[(r, "iso")] = (wi, a); params[(r, "run")] = (wr, a)
    return params, pd.DataFrame(log_rows)


def gpu_predict(bst, X, it):
    """Predict on the GPU (cupy) when possible; CPU-side DMatrix prediction was ~10 min per model."""
    try:
        import cupy as cp
        out = []
        for i in range(0, len(X), 1_000_000):
            out.append(cp.asnumpy(bst.inplace_predict(cp.asarray(X[i:i + 1_000_000]), iteration_range=it)))
        return np.concatenate(out)
    except Exception as ex:
        log("  cupy prediction unavailable:", repr(ex)[:120])
        out = []
        for i in range(0, len(X), 1_000_000):
            out.append(bst.predict(xgb.DMatrix(X[i:i + 1_000_000]), iteration_range=it))
        return np.concatenate(out)
# ----------------------------------------------------------------------------------- U-Net (our own)
# A 2-D U-Net over (time x station) day grids. Input = released anchors (as deviations from the LODO profile),
# quality masks, profiles, ramps, static link data; output = speed / flow-per-lane / density-per-lane deviations.
# Trained on train days with the official masks plus random extra anchor dropout (targets known on train).
import torch, torch.nn as nn, torch.nn.functional as TF

UCFG = dict(epochs=int(os.environ.get("U_EPOCHS", 60)), batch=8, lr=2e-3, wd=1e-4, ch=int(os.environ.get("U_CH", 64)),
            drop_max=0.35, seed=int(os.environ.get("U_SEED", 2026)), lam_k=4.0, lam_vq=float(os.environ.get("U_LAMVQ", 1.0)),
            q_w=1.0 / 512)
torch.manual_seed(UCFG["seed"])
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SV, SQ, SK = 20.0, 400.0, 15.0          # normalisation of speed / flow-per-lane / density-per-lane deviations


def unet_arrays(g, pn, with_truth):
    """Per-split float16 channel stack (C, T, S); anchor mask; target mask; (truth targets)."""
    V, Q, DN = g["v"], g["q"], g["dn"]
    lanes = pn.lanes[None, :].astype(np.float32)
    fin = np.isfinite(V) & np.isfinite(Q)
    A = fin & g["elig"] & ~g["TG"]
    AQ = fin & ~g["elig"]
    Pv, Pq, Pk = g["Pv"], g["Pq"], g["Pk"]
    T, S = V.shape
    tod = (np.arange(T) % 288) / 288.0 * 2 * np.pi
    z = lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    chans = [
        z((V - Pv) / SV), z((Q - Pq) / lanes / SQ), z((DN - Pk) / lanes / SK),        # 0-2 observed devs (masked by A on the fly)
        AQ.astype(np.float32), z(np.where(AQ, (V - Pv) / SV, 0)), z(np.where(AQ, (Q - Pq) / lanes / SQ, 0)),
        Pv / 100.0 - 1.0, Pq / lanes / 1500.0 - 1.0, Pk / lanes / 30.0 - 1.0,
        g["on_f"] / lanes / 100.0, g["off_f"] / lanes / 100.0,
        np.broadcast_to(pn.lanes[None, :] / 5.0 - 1.0, (T, S)), np.broadcast_to(pn.s_vf[None, :] / 115.0 - 1.0, (T, S)),
        np.broadcast_to(np.log(np.maximum(pn.s_len, 0.005))[None, :] / 3.0 + 0.5, (T, S)),
        np.broadcast_to(pn.s_capl[None, :] / 2000.0 - 1.0, (T, S)),
        np.broadcast_to(np.sin(tod)[:, None], (T, S)), np.broadcast_to(np.cos(tod)[:, None], (T, S)),
        np.broadcast_to(g["rate_row"][:, None], (T, S)),
    ]
    X = np.stack([np.asarray(c, np.float32) for c in chans]).astype(np.float16)
    out = dict(X=X, A=A, TG=g["TG"].copy())
    if with_truth:
        out["P"] = np.stack([Pv, Pq, Pk]).astype(np.float32)
    if with_truth:
        Et = g["Et"]
        Vt, Qt = g["Vt"], g["Qt"]
        Kt = Qt / np.maximum(Vt, 1.0)
        out["Y"] = np.stack([z((Vt - Pv) / SV), z((Qt - Pq) / lanes / SQ), z((Kt - Pk) / lanes / SK)]).astype(np.float16)
        out["Et"] = Et
    return out


U_INTERP = int(os.environ.get("U_INTERP", 0))
N_IN = 18 + 2 + (5 if U_INTERP else 0)   # 18 stacked channels + anchor flag + target flag (+ GPU interpolation)


def interp_channels(Xobs, A, dec=12.0):
    """Per-station temporal interpolation of the anchor deviations (computed after anchor dropout, so no leak)."""
    B, C, T, S = Xobs.shape
    t = torch.arange(T, device=Xobs.device).view(1, T, 1).expand(B, T, S)
    prev = torch.cummax(torch.where(A, t, torch.full_like(t, -1)), dim=1).values
    nxt = torch.flip(torch.cummin(torch.flip(torch.where(A, t, torch.full_like(t, T)), [1]), dim=1).values, [1])
    hp = prev >= 0; hn = nxt < T; both = hp & hn
    pp = prev.clamp(min=0); nn_ = nxt.clamp(max=T - 1)
    g1 = (t - pp).float(); g2 = (nn_ - t).float()
    w = torch.where(both, g1 / (g1 + g2).clamp(min=1.0), torch.zeros_like(g1))
    out = []
    for c in range(C):
        x = Xobs[:, c]
        xp = torch.gather(x, 1, pp); xn = torch.gather(x, 1, nn_)
        one = torch.where(hp, xp * torch.exp(-g1 / dec), torch.where(hn, xn * torch.exp(-g2 / dec), torch.zeros_like(xp)))
        out.append(torch.where(both, xp * (1 - w) + xn * w, one))
    out.append(torch.where(hp, g1.clamp(max=48.0) / 48.0, torch.ones_like(g1)))
    out.append(torch.where(hn, g2.clamp(max=48.0) / 48.0, torch.ones_like(g2)))
    return torch.stack(out, 1)


class Block(nn.Module):
    def __init__(s, cin, cout):
        super().__init__()
        s.c1 = nn.Conv2d(cin, cout, 3, padding=1); s.n1 = nn.GroupNorm(8, cout)
        s.c2 = nn.Conv2d(cout, cout, 3, padding=1); s.n2 = nn.GroupNorm(8, cout)

    def forward(s, x):
        x = TF.gelu(s.n1(s.c1(x)))
        return TF.gelu(s.n2(s.c2(x)))


class UNet(nn.Module):
    def __init__(s, cin=N_IN, ch=48, cout=3):
        super().__init__()
        s.e0 = Block(cin, ch); s.e1 = Block(ch, 2 * ch); s.e2 = Block(2 * ch, 4 * ch); s.e3 = Block(4 * ch, 4 * ch)
        s.u2 = Block(8 * ch, 2 * ch); s.u1 = Block(4 * ch, ch); s.u0 = Block(2 * ch, ch)
        s.out = nn.Conv2d(ch, cout, 1)

    def forward(s, x):
        x0 = s.e0(x)
        x1 = s.e1(TF.max_pool2d(x0, 2, ceil_mode=True))
        x2 = s.e2(TF.max_pool2d(x1, 2, ceil_mode=True))
        x3 = s.e3(TF.max_pool2d(x2, 2, ceil_mode=True))
        y2 = s.u2(torch.cat([TF.interpolate(x3, size=x2.shape[-2:], mode="nearest"), x2], 1))
        y1 = s.u1(torch.cat([TF.interpolate(y2, size=x1.shape[-2:], mode="nearest"), x1], 1))
        y0 = s.u0(torch.cat([TF.interpolate(y1, size=x0.shape[-2:], mode="nearest"), x0], 1))
        return s.out(y0)


def make_batch(ua, days, drop, gen):
    """Assemble a batch of whole days from one panel's arrays (on GPU)."""
    sl = [slice(d * 288, (d + 1) * 288) for d in days]
    X = torch.from_numpy(np.stack([ua["X"][:, s_] for s_ in sl])).to(DEV).float()
    A = torch.from_numpy(np.stack([ua["A"][s_] for s_ in sl])).to(DEV)
    TGm = torch.from_numpy(np.stack([ua["TG"][s_] for s_ in sl])).to(DEV)
    if drop > 0:
        extra = A & (torch.rand(A.shape, device=DEV, generator=gen) < drop)
        A = A & ~extra
        TGm = TGm | extra
    Af = A.float().unsqueeze(1)
    obs = X[:, :3] * Af
    parts = [obs, X[:, 3:], Af, TGm.float().unsqueeze(1)]
    if U_INTERP:
        parts.append(interp_channels(obs, A))
    X = torch.cat(parts, 1)
    return X, TGm, sl


def unet_loss(out, ua, sl, TGm, pn):
    Y = torch.from_numpy(np.stack([ua["Y"][:, s_] for s_ in sl])).to(DEV).float()
    Et = torch.from_numpy(np.stack([ua["Et"][s_] for s_ in sl])).to(DEV)
    P3 = torch.from_numpy(np.stack([ua["P"][:, s_] for s_ in sl])).to(DEV).float()   # profiles v, q, k (natural units)
    m = (TGm & Et).float()
    n = m.sum().clamp(min=1.0)
    lanes = torch.as_tensor(pn.lanes, device=DEV, dtype=torch.float32)[None, None, :]
    L = torch.as_tensor(pn.s_len, device=DEV, dtype=torch.float32)[None, None, :]
    ev = (out[:, 0] - Y[:, 0]) * SV
    eq = (out[:, 1] - Y[:, 1]) * SQ
    ek = (out[:, 2] - Y[:, 2]) * SK * lanes * L        # vehicles on the link
    lv = (ev * ev * m).sum() / n; lq = (eq * eq * m).sum() / n; lk = (ek.abs() * m).sum() / n
    loss = lv + UCFG["q_w"] * lq + UCFG["lam_k"] * lk
    lvq = torch.zeros((), device=DEV)
    if UCFG["lam_vq"] > 0:                             # density implied by the speed/flow heads (Task 3 consistency)
        vp = (P3[:, 0] + out[:, 0] * SV).clamp(min=3.0); qp = (P3[:, 1] + out[:, 1] * SQ * lanes).clamp(min=0.0)
        vt = P3[:, 0] + Y[:, 0] * SV; qt = P3[:, 1] + Y[:, 1] * SQ * lanes
        en = (qp / vp - qt / vt.clamp(min=1.0)) * L
        lvq = (en.abs() * m).sum() / n
        loss = loss + UCFG["lam_vq"] * lvq
    return loss, (lv.item(), lq.item(), lk.item(), lvq.item())


def unet_predict(model, ua, days):
    model.eval()
    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
        for i in range(0, len(days), 8):
            X, TGm, sl = make_batch(ua, days[i:i + 8], 0.0, None)
            preds.append(model(X).float().cpu().numpy())
    P = np.concatenate(preds)                      # (D, 3, 288, S)
    return np.concatenate([p for p in P], axis=1)  # (3, D*288, S)


def main():
    rng = np.random.default_rng(UCFG["seed"])
    panels, UA = [], {}
    for pi, P in enumerate(PANELS):
        pn = load_panel(pi, P)
        build_profiles(pn)
        for split in ("train", "validation", "private"):
            UA[(pi, split)] = unet_arrays(pn.g[split], pn, split == "train")
        pn.ph = physics_setup(pn)
        g = pn.g["train"]
        off = int(np.where(g["times"] >= CFG["holdout_from"])[0][0])
        sel = (g["tr"] >= off) & g["Et"][g["tr"], g["tc"]]
        pn.hold = dict(off=off, tr=g["tr"][sel], tc=g["tc"][sel], reg=g["reg"][off:],
                       Vt=g["Vt"][off:].copy(), Qt=g["Qt"][off:].copy(), Pv=g["Pv"][off:].copy(),
                       Pq=g["Pq"][off:].copy(), Pk=g["Pk"][off:].copy())
        for split in ("validation", "private"):
            gs = pn.g[split]
            pn.g[split] = {k: gs[k] for k in ("times", "reg", "tr", "tc", "Pv", "Pq", "Pk")}
        pn.g["train"] = None
        os.makedirs(f"{OUT}/panels", exist_ok=True)
        save_panel_meta(pn, f"{OUT}/panels/{P}.npz")
        panels.append(pn); gc.collect()
        log(f"unet arrays {P}")
    ndays = {pn.pi: len(UA[(pn.pi, "train")]["A"]) // 288 for pn in panels}
    tdays = pd.date_range("2030-06-01", "2031-03-01", freq="D", inclusive="left").strftime("%Y-%m-%d").to_numpy()
    fit_days = np.where(tdays < CFG["es_from"])[0]
    es_days = np.where((tdays >= CFG["es_from"]) & (tdays < CFG["holdout_from"]))[0]
    hold_days = np.where(tdays >= CFG["holdout_from"])[0]
    FULL = os.environ.get("U_MODE", "holdout") == "full"
    if FULL:            # final model: every train day is fitted, fixed schedule, last epoch kept
        fit_days = np.arange(len(tdays))
        log("U_MODE=full: fitting all", len(fit_days), "train days; held-out numbers below are in-sample")
    model = UNet(ch=UCFG["ch"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=UCFG["lr"], weight_decay=UCFG["wd"])
    nb_epoch = sum(len(fit_days) // UCFG["batch"] for _ in panels)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=UCFG["lr"], total_steps=UCFG["epochs"] * nb_epoch, pct_start=0.1)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")
    gen = torch.Generator(device=DEV); gen.manual_seed(UCFG["seed"])
    best, best_state = 1e18, None
    for ep in range(UCFG["epochs"]):
        model.train(); t1 = time.time()
        jobs = []
        for pn in panels:
            d = rng.permutation(fit_days)
            jobs += [(pn.pi, d[i:i + UCFG["batch"]]) for i in range(0, len(d) - UCFG["batch"] + 1, UCFG["batch"])]
        order = rng.permutation(len(jobs))
        tot = np.zeros(4); nbt = 0
        for j in order:
            pi, days = jobs[j]; pn = panels[pi]; ua = UA[(pi, "train")]
            drop = float(rng.uniform(0, UCFG["drop_max"]))
            X, TGm, sl = make_batch(ua, days, drop, gen)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                out = model(X)
            loss, parts = unet_loss(out.float(), ua, sl, TGm, pn)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += parts; nbt += 1
        # early-stopping days (official masks only)
        model.eval(); vt = np.zeros(4); nv = 0
        with torch.no_grad():
            for pn in panels:
                ua = UA[(pn.pi, "train")]
                for i in range(0, len(es_days), 8):
                    X, TGm, sl = make_batch(ua, es_days[i:i + 8], 0.0, None)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                        out = model(X)
                    _, parts = unet_loss(out.float(), ua, sl, TGm, pn)
                    vt += parts; nv += 1
        vt /= max(nv, 1); score = vt[0] + UCFG["q_w"] * vt[1] + UCFG["lam_k"] * vt[2] + UCFG["lam_vq"] * vt[3]
        log(f"epoch {ep}: train mse_v {tot[0]/nbt:.3f} mse_q {tot[1]/nbt:.1f} mae_N {tot[2]/nbt:.4f} mae_Nvq {tot[3]/nbt:.4f} | "
            f"es mse_v {vt[0]:.3f} mse_q {vt[1]:.1f} mae_N {vt[2]:.4f} mae_Nvq {vt[3]:.4f} score {score:.4f} ({time.time() - t1:.0f}s)")
        if score < best or FULL:
            best = score; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if time.time() - T0 > 4.0 * 3600:
            log("time budget reached; stopping training"); break
    model.load_state_dict(best_state)
    torch.save(best_state, f"{OUT}/unet.pt")
    # predictions on held-out February and the two test months
    rows = []
    for pn in panels:
        h = pn.hold
        P_ = unet_predict(model, UA[(pn.pi, "train")], hold_days)
        off = h["off"]
        assert off == hold_days[0] * 288
        r_, c_ = h["tr"], h["tc"]
        lanes = pn.lanes[c_]
        v = h["Pv"][r_ - off, c_] + P_[0, r_ - off, c_] * SV
        q = h["Pq"][r_ - off, c_] + P_[1, r_ - off, c_] * SQ * lanes
        k = h["Pk"][r_ - off, c_] + P_[2, r_ - off, c_] * SK * lanes
        rows.append(pd.DataFrame(dict(pi=pn.pi, split="hold", row=r_, col=c_, uv=v, uq=q, uk=k)))
        for split in ("validation", "private"):
            gs = pn.g[split]; ua = UA[(pn.pi, split)]
            nd = len(ua["A"]) // 288
            P_ = unet_predict(model, ua, np.arange(nd))
            r_, c_ = gs["tr"], gs["tc"]; lanes = pn.lanes[c_]
            v = gs["Pv"][r_, c_] + P_[0, r_, c_] * SV
            q = gs["Pq"][r_, c_] + P_[1, r_, c_] * SQ * lanes
            k = gs["Pk"][r_, c_] + P_[2, r_, c_] * SK * lanes
            rows.append(pd.DataFrame(dict(pi=pn.pi, split=split, row=r_, col=c_, uv=v, uq=q, uk=k)))
    U = pd.concat(rows, ignore_index=True)
    if FULL:   # in-sample on February: only the test rows are usable downstream
        U[U.split != "hold"].to_parquet(f"{OUT}/unet_full_preds.parquet", index=False)
    else:
        U.to_parquet(f"{OUT}/unet_preds.parquet", index=False)
    # quick held-out S_state
    res = []
    for pn in panels:
        h = pn.hold; u = U[(U.pi == pn.pi) & (U.split == "hold")]
        r_, c_ = u.row.to_numpy() - h["off"], u.col.to_numpy()
        ss = state_score(pn, c_, h["reg"][r_], np.clip(u.uv.to_numpy(), 3, 140), np.clip(u.uq.to_numpy(), 0, None), h["Vt"][r_, c_], h["Qt"][r_, c_])
        for r, d in ss.items():
            res.append(dict(panel=pn.P, regime=r, **d))
    res = pd.DataFrame(res)
    log("U-Net held-out:\n" + res.groupby("regime")[["rmse_v", "rmse_q", "S_state"]].mean().to_string())
    log(f"U-Net held-out S_state (panel macro) {res.groupby('panel').S_state.mean().mean():.5f}")
    json.dump(dict(ucfg=UCFG, interp=U_INTERP, mode="full" if FULL else "holdout", best_es=float(best),
                   heldout=res.to_dict(orient="records")), open(f"{OUT}/unet_metrics.json", "w"), indent=1, default=float)
    log("done")




# ----------------------------------------------------------------------------------- U-Net v3: no memory of the train run
# The scored months come from their own simulation run (other bottlenecks). Task 1 is offline, so a month's own released cells may be used for that month.
# Profiles: month-own, leave-one-day-out, weekday/weekend + all days, released cells only, in every month (train months too, so all months look alike).
# Training: train split days (official targets + extra dropout) + March and April days (extra dropout on released cells; official targets have no truth, no loss there).
# 20% of the released March and April cells are held out from everything (fixed); they measure the model. Final predictions use all released cells.
KAPPA = 6.0
HOLD = 0.20

def own_profile(g, A, times):
    V, Q = g["v"], g["q"]; T, S = V.shape
    K = Q / np.maximum(V, 1.0)
    out = {n: np.zeros((T, S), np.float32) for n in ("v", "q", "k")}
    months = pd.Series(pd.Index(times).str[:7])
    for m, idx in months.groupby(months).groups.items():
        a, b = int(idx[0]), int(idx[-1]) + 1; nd = (b - a) // 288
        wk = np.array([pd.Timestamp(t).weekday() >= 5 for t in times[a:b:288]])
        c = A[a:b].reshape(nd, 288, S).astype(np.float64)
        sm = lambda z: z + np.roll(z, 1, axis=1) + np.roll(z, -1, axis=1)
        c3 = sm(c); Cn = c3.sum(0); Cc = c.sum((0, 1))
        for n, X in (("v", V), ("q", Q), ("k", K)):
            x = np.where(A[a:b], X[a:b], 0.0).reshape(nd, 288, S).astype(np.float64)
            x3 = sm(x); Xn = x3.sum(0); Xc = x.sum((0, 1))
            Xt = {t_: x3[wk == t_].sum(0) for t_ in (False, True)}; Ct = {t_: c3[wk == t_].sum(0) for t_ in (False, True)}
            res = np.zeros((nd, 288, S), np.float32)
            for d in range(nd):
                lm = (Xc - x[d].sum(0)) / np.maximum(Cc - c[d].sum(0), 1.0)                  # the link's mean on the other days of the month
                na, da = Xn - x3[d], Cn - c3[d]
                Pall = np.where(da > 0, na / np.maximum(da, 1e-9), lm[None, :])
                nt, dt = Xt[wk[d]] - x3[d], Ct[wk[d]] - c3[d]
                res[d] = (nt + KAPPA * Pall) / (dt + KAPPA)
            out[n][a:b] = res.reshape(b - a, S)
    return out["v"], out["q"], out["k"]

def scored_arrays(g, pn, A_use, TG_use):
    Pv, Pq, Pk = own_profile(g, A_use, g["times"])
    g2 = dict(g); g2["Pv"], g2["Pq"], g2["Pk"] = Pv, Pq, Pk
    ua = unet_arrays(g2, pn, False)
    lanes = pn.lanes[None, :].astype(np.float32)
    z = lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    V, Q = g["v"], g["q"]; K = Q / np.maximum(V, 1.0)
    ua["Y"] = np.stack([z((V - Pv) / SV), z((Q - Pq) / lanes / SQ), z((K - Pk) / lanes / SK)]).astype(np.float16)
    ua["P"] = np.stack([Pv, Pq, Pk]).astype(np.float32); ua["Et"] = A_use; ua["A"] = A_use; ua["TG"] = TG_use
    return ua

def predict_days(model, ua, nd):
    return unet_predict(model, ua, np.arange(nd))

def main3():
    rng = np.random.default_rng(UCFG["seed"]); hold_rng = np.random.default_rng(777)
    panels = []; UA = {}; HO = {}; FIN = {}
    for pi, P in enumerate(PANELS):
        pn = load_panel(pi, P)
        g = pn.g["train"]
        Atr = np.isfinite(g["v"]) & np.isfinite(g["q"]) & g["elig"] & ~g["TG"]
        g["Pv"], g["Pq"], g["Pk"] = own_profile(g, Atr, g["times"])
        UA[(pi, "train")] = unet_arrays(g, pn, True)
        for split in ("validation", "private"):
            gs = pn.g[split]
            A = np.isfinite(gs["v"]) & np.isfinite(gs["q"]) & gs["elig"] & ~gs["TG"]
            ex = A & (hold_rng.random(A.shape) < HOLD)
            UA[(pi, split)] = scored_arrays(gs, pn, A & ~ex, gs["TG"] | ex)
            HO[(pi, split)] = dict(ex=ex, V=gs["v"].copy(), Q=gs["q"].copy(), reg=gs["reg"].copy())
            FIN[(pi, split)] = dict(g=dict(gs), A=A, tr=gs["tr"], tc=gs["tc"])
        pn.g["train"] = None
        panels.append(pn); gc.collect()
        log(f"arrays {P}")
    tdays = pd.date_range("2030-06-01", "2031-03-01", freq="D", inclusive="left").strftime("%Y-%m-%d").to_numpy()
    nday = {s_: len(UA[(0, s_)]["A"]) // 288 for s_ in ("train", "validation", "private")}
    model = UNet(ch=UCFG["ch"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=UCFG["lr"], weight_decay=UCFG["wd"])
    B = UCFG["batch"]; EPOCHS = int(os.environ.get("U_EPOCHS3", 45))
    def epoch_jobs():
        jobs = []
        for pn in panels:
            d = rng.permutation(nday["train"]); jobs += [(pn.pi, "train", d[i:i + B]) for i in range(0, len(d) - B + 1, B)]
            for s_ in ("validation", "private"):
                for _ in range(2):      # scored days count double
                    d = rng.permutation(nday[s_]); jobs += [(pn.pi, s_, d[i:i + B]) for i in range(0, len(d) - B + 1, B)]
        return jobs
    nb = len(epoch_jobs())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=UCFG["lr"], total_steps=EPOCHS * nb, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    gen = torch.Generator(device=DEV); gen.manual_seed(UCFG["seed"])
    def harness(tag):
        rows = []; w_rows = []
        model.eval()
        for pn in panels:
            for s_ in ("validation", "private"):
                ua = UA[(pn.pi, s_)]; h = HO[(pn.pi, s_)]; nd = nday[s_]
                O = predict_days(model, ua, nd)
                lanes = pn.lanes[None, :]
                vp = np.clip(ua["P"][0] + O[0] * SV, 3, 140); qp = np.clip(ua["P"][1] + O[1] * SQ * lanes, 0, None); kp = np.clip(ua["P"][2] + O[2] * SK * lanes, .05, None)
                ex = h["ex"]; ev = (vp - h["V"])[ex]; eq = ((qp - h["Q"]) / lanes)[ex]
                rv, rq = float(np.sqrt(np.mean(ev ** 2))), float(np.sqrt(np.mean(eq ** 2)))
                rows.append(dict(panel=pn.P, split=s_, rmse_v=rv, rmse_q=rq, s_state=.54 * max(0, 1 - rv / 25) + .46 * max(0, 1 - rq / 600), n=int(ex.sum())))
                # conservation proxy on observed transitions, plain and with the density head folded in
                known = ua["A"] | ex; L_ = pn.s_len[None, :]
                Nt = np.where(known, h["Q"] / np.maximum(h["V"], 1.0), np.nan) * L_
                tr_ = known[1:] & known[:-1]; dNt = (Nt[1:] - Nt[:-1])
                for w in (0.0, .25, .5, .625, .75, 1.0):
                    va, qa = adjust(vp, qp, kp, w, 1.0)
                    e = np.where(ex, qa / np.maximum(va, 1.0) * L_ - Nt, 0.0)
                    de = e[1:] - e[:-1]
                    ok = tr_ & ((ex[1:] | ex[:-1]))
                    w_rows.append(dict(panel=pn.P, split=s_, w=w, num=np.nansum(np.abs(de)[tr_]), den=np.nansum(np.abs(dNt)[tr_]), rq=float(np.sqrt(np.mean(((qa - h["Q"]) / lanes)[ex] ** 2))), rv=float(np.sqrt(np.mean((va - h["V"])[ex] ** 2)))))
        R_ = pd.DataFrame(rows); W_ = pd.DataFrame(w_rows); W_["E"] = W_.num / W_.den
        log(tag, "held-out released cells: mean over panels\n" + R_.groupby("split")[["rmse_v", "rmse_q", "s_state"]].mean().round(4).to_string())
        log(tag, "density head weight: conservation proxy E, RMSE\n" + W_.groupby(["split", "w"])[["E", "rv", "rq"]].mean().round(4).to_string())
        return R_
    for ep in range(EPOCHS):
        model.train(); t1 = time.time(); jobs = epoch_jobs(); tot = np.zeros(4)
        for j in rng.permutation(len(jobs)):
            pi, s_, days = jobs[j]; pn = panels[pi]; ua = UA[(pi, s_)]
            X, TGm, sl = make_batch(ua, days, float(rng.uniform(0, UCFG["drop_max"])), gen)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                out = model(X)
            loss, parts = unet_loss(out.float(), ua, sl, TGm, pn)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tot += parts
        log(f"epoch {ep + 1}/{EPOCHS}: mse_v {tot[0] / len(jobs):.3f} mse_q {tot[1] / len(jobs):.1f} mae_N {tot[2] / len(jobs):.4f} ({time.time() - t1:.0f}s)")
        if (ep + 1) in (15, 30):
            harness(f"epoch {ep + 1}")
    torch.save(model.state_dict(), f"{OUT}/unet3.pt")
    R_ = harness("final"); R_.to_csv(f"{OUT}/harness.csv", index=False)
    log("per panel\n" + R_.pivot_table(index="panel", columns="split", values=["s_state", "rmse_v"]).round(4).to_string())
    # official targets, all released cells as inputs
    rows = []
    for pn in panels:
        for s_ in ("validation", "private"):
            f = FIN[(pn.pi, s_)]; gs = f["g"]
            ua = scored_arrays(gs, pn, f["A"], gs["TG"])
            O = predict_days(model, ua, nday[s_]); r_, c_ = f["tr"], f["tc"]; lanes = pn.lanes[c_]
            rows.append(pd.DataFrame(dict(pi=pn.pi, split=s_, row=r_, col=c_, uv=ua["P"][0][r_, c_] + O[0, r_, c_] * SV, uq=ua["P"][1][r_, c_] + O[1, r_, c_] * SQ * lanes,
                                          uk=ua["P"][2][r_, c_] + O[2, r_, c_] * SK * lanes, reg=gs["reg"][r_], key=pd.Series(gs["times"][r_]).radd(f"{s_}|{pn.P}|").to_numpy() + "|" + pn.st[c_] + "|" + pn.st_link[c_] + "|" + gs["reg"][r_])))
        log("final predictions", pn.P)
    U = pd.concat(rows, ignore_index=True)
    assert np.isfinite(U[["uv", "uq", "uk"]].to_numpy()).all()
    U.to_parquet(f"{OUT}/unet3_preds.parquet", index=False)
    log("done", len(U))
main3()
