"""Stage 01 - data preparation for Task 2: dense time x link arrays per corridor and split (speed, flow, occupancy, eligibility, ramps,
train masked-layer availability) and the published window tables, written to <TFB_OUT>/prep. Train arrays use the unmasked layer;
validation/private arrays use the masked layer with each scored window's 60-minute history overlaid from window_history."""
# ---- paths: Kaggle by default; run_all.py points these at local folders -------------------------------------
import os as _os, glob as _glob
TFB_ROOTS = [r for r in _os.environ.get("TFB_INPUT", "/kaggle/input").split(_os.pathsep) if r]
TFB_OUT = _os.environ.get("TFB_OUT", "/kaggle/working"); _os.makedirs(TFB_OUT, exist_ok=True)
def G(pat):
    """every file matching pat under the input roots (recursive), sorted"""
    return sorted(set(p for r in TFB_ROOTS for p in _glob.glob(_os.path.join(r, "**", pat), recursive=True)))
# ----------------------------------------------------------------------------------------------------------------
import os, sys, glob, json, time, math, gc, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

def _run_prep():
    # TFB Task 2 - data preparation: dense per-panel arrays (time x link) for every split.
    # Output (/kaggle/working/prep/<panel>.npz + meta.json) is attached to later kernels as a kernel source.
    import os, glob, json, time
    import numpy as np, pandas as pd, pyarrow.parquet as pq

    T0 = time.time()
    def el(): return f"[{time.time()-T0:.0f}s]"
    DATA = G("kaggle_public")[0]
    OUT = _os.path.join(TFB_OUT, "prep"); os.makedirs(OUT, exist_ok=True)
    PANELS = [p["corridor_id"] for p in json.load(open(f"{DATA}/config/corridors.json"))["panels"]]
    Q2 = [p for p in PANELS if os.path.isdir(f"{DATA}/task2/{p}")]
    STEP = pd.Timedelta(minutes=5)

    def split_days(split):
        """first day and number of days of a split, read from the timestamps of its state files (a regenerated month may have other dates)"""
        layer = "mainline_states" if split == "train" else "mainline_states_masked"
        files = sorted(glob.glob(f"{DATA}/corridors/{Q2[0]}/{split}/{layer}/**/*.parquet", recursive=True))
        if not files:
            raise SystemExit(f"no {layer} files for split '{split}' under {DATA}/corridors/{Q2[0]}")
        lo = hi = None
        for f in files:
            t = pd.to_datetime(pq.ParquetFile(f).read(columns=["timestamp"]).column(0).to_pandas(), utc=True)
            lo = t.min() if lo is None else min(lo, t.min()); hi = t.max() if hi is None else max(hi, t.max())
        d0 = lo.floor("D")
        return d0.strftime("%Y-%m-%d"), int((hi.floor("D") - d0) / pd.Timedelta(days=1)) + 1

    SPLITS = {s: split_days(s) for s in ("train", "validation", "private")}
    print("splits (first day, days):", SPLITS, el(), flush=True)     # released data: train 2030-06-01 273, validation 2031-03-01 31, private 2031-04-01 30


    def network(p):
        net = f"{DATA}/corridors/{p}/network"
        lk = pd.read_csv(f"{net}/links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
        fd = pd.read_csv(f"{net}/fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
        tp = pd.read_csv(f"{net}/lwr_mainline_topology.csv", dtype=str)
        tp["order_index"] = tp.order_index.astype(int)
        tp = tp.sort_values("order_index")
        order_idx = tp.link_id.astype(str).tolist()
        mainline = set(order_idx)
        split_ids = lambda v: [x.strip() for x in str(v).split(";") if x.strip() in mainline] if str(v) != "nan" else []
        out_main = {l: split_ids(o) for l, o in zip(order_idx, tp.outgoing_link_ids)}
        in_main = {l: split_ids(i) for l, i in zip(order_idx, tp.incoming_link_ids)}
        heads = [l for l in order_idx if not in_main[l]]
        multi_out = [l for l in order_idx if len(out_main[l]) > 1]
        chain, seen = [], set()
        if len(heads) == 1:
            cur = heads[0]
            while cur is not None and cur not in seen:
                chain.append(cur); seen.add(cur)
                nx = [x for x in out_main[cur] if x not in seen]
                cur = nx[0] if nx else None
        mism = [(a, out_main[a], b) for a, b in zip(order_idx[:-1], order_idx[1:]) if b not in out_main[a]]
        print(p, "topology: heads", len(heads), heads[:5], "multi-out", len(multi_out), "chain len", len(chain), "of", len(order_idx),
              "order_index mismatches", len(mism), mism[:6], flush=True)
        order = chain if len(chain) == len(order_idx) else order_idx
        assert set(order) == set(lk.index), (p, len(order), len(lk))
        idx = {l: i for i, l in enumerate(order)}
        nxt_ok = [b in out_main[a] for a, b in zip(order[:-1], order[1:])]
        ra = pd.read_csv(f"{net}/ramp_attachment_map.csv", dtype=str)
        info = dict(
            links=order,
            vf=lk.loc[order, "free_speed_kmh"].to_numpy(np.float32),
            lanes=lk.loc[order, "lanes"].to_numpy(np.float32),
            cap=lk.loc[order, "capacity_vph"].to_numpy(np.float32),
            length=lk.loc[order, "length_km"].to_numpy(np.float32),
            kc=fd.reindex(order)["critical_density"].to_numpy(np.float32),
            kj=fd.reindex(order)["k_jam"].to_numpy(np.float32),
            ramps=ra.ramp_link_id.tolist(),
            ramp_type=ra.ramp_type.tolist(),
            ramp_link=np.array([idx.get(x, -1) for x in ra.nearest_mainline_link_id], np.int32),
            n_on_topo=np.array([0 if str(x) == "nan" else len(str(x).split(";")) for x in tp.set_index("link_id").reindex(order).on_ramp_link_ids], np.int16),
            n_off_topo=np.array([0 if str(x) == "nan" else len(str(x).split(";")) for x in tp.set_index("link_id").reindex(order).off_ramp_link_ids], np.int16),
        )
        print(p, "links", len(order), "chain ok frac", np.mean(nxt_ok).round(4), "ramps", len(ra), dict(pd.Series(info["ramp_type"]).value_counts()),
              "unattached ramps", int((info["ramp_link"] < 0).sum()), flush=True)
        return info


    def read_mainline(p, split, layer, links):
        start, ndays = SPLITS[split]
        t0 = pd.Timestamp(start, tz="UTC")
        n = ndays * 288
        L = len(links)
        idx = {l: i for i, l in enumerate(links)}
        V = np.full((n, L), np.nan, np.float32); Q = V.copy(); O = V.copy()
        E = np.zeros((n, L), np.int8)       # is_score_eligible
        files = sorted(glob.glob(f"{DATA}/corridors/{p}/{split}/{layer}/**/*.parquet", recursive=True))
        cols = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "is_score_eligible"]
        dup_total = 0
        for f in files:
            t = pq.ParquetFile(f).read(columns=cols).to_pandas()
            ut, it = np.unique(t.timestamp.values, return_inverse=True)
            steps = ((pd.to_datetime(ut, utc=True) - t0) // STEP).to_numpy().astype(np.int64)
            r = steps[it]
            ul, il = np.unique(t.link_id.astype(str).values, return_inverse=True)
            cmap = np.array([idx.get(x, -1) for x in ul], np.int64)
            c = cmap[il]
            ok = (c >= 0) & (r >= 0) & (r < n)
            r, c = r[ok], c[ok]
            sp = t.speed_kmh.to_numpy(np.float32)[ok]; fl = t.flow_vph.to_numpy(np.float32)[ok]
            oc = t.occupancy.to_numpy(np.float32)[ok]; el_ = t.is_score_eligible.to_numpy()[ok].astype(np.int8)
            key = r * L + c
            if len(np.unique(key)) != len(key):
                # several stations on one link: average the non-null values, eligible if any eligible
                dup_total += len(key) - len(np.unique(key))
                df = pd.DataFrame({"k": key, "v": sp, "q": fl, "o": oc, "e": el_})
                g = df.groupby("k").agg(v=("v", "mean"), q=("q", "mean"), o=("o", "mean"), e=("e", "max"))
                k = g.index.to_numpy()
                r, c = k // L, k % L
                sp, fl, oc, el_ = g.v.to_numpy(np.float32), g.q.to_numpy(np.float32), g.o.to_numpy(np.float32), g.e.to_numpy(np.int8)
            V[r, c] = sp; Q[r, c] = fl; O[r, c] = oc; E[r, c] = el_
        print(f"  {p} {split} {layer}: files {len(files)} cells {n}x{L} nonnull {np.isfinite(V).mean():.3f} elig {E.mean():.3f} dup {dup_total}", el(), flush=True)
        return V, Q, O, E


    def read_ramps(p, split, ramps):
        start, ndays = SPLITS[split]
        t0 = pd.Timestamp(start, tz="UTC"); n = ndays * 288
        idx = {r: i for i, r in enumerate(ramps)}
        R = np.full((n, len(ramps)), np.nan, np.float32); RE = np.zeros((n, len(ramps)), np.int8)
        files = sorted(glob.glob(f"{DATA}/corridors/{p}/{split}/ramp_states/**/*.parquet", recursive=True))
        for f in files:
            t = pq.ParquetFile(f).read(columns=["timestamp", "ramp_link_id", "flow_vph", "is_score_eligible"]).to_pandas()
            ut, it = np.unique(t.timestamp.values, return_inverse=True)
            steps = ((pd.to_datetime(ut, utc=True) - t0) // STEP).to_numpy().astype(np.int64)
            r = steps[it]
            c = t.ramp_link_id.map(idx).fillna(-1).astype(np.int64).to_numpy()
            ok = (c >= 0) & (r >= 0) & (r < n)
            R[r[ok], c[ok]] = t.flow_vph.to_numpy(np.float32)[ok]; RE[r[ok], c[ok]] = t.is_score_eligible.to_numpy()[ok].astype(np.int8)
        print(f"  {p} {split} ramps: files {len(files)} nonnull {np.isfinite(R).mean():.3f}", el(), flush=True)
        return R, RE


    def windows(p, split):
        w = pd.read_csv(f"{DATA}/task2/{p}/{split}/window_index.csv")
        t0 = pd.Timestamp(SPLITS[split][0], tz="UTC")
        w["t_idx"] = ((pd.to_datetime(w.forecast_origin, utc=True) - t0) // STEP).astype(int)
        return w


    meta = {"panels": Q2, "splits": SPLITS}
    for p in Q2:
        info = network(p)
        arrs = {k: v for k, v in info.items() if isinstance(v, np.ndarray)}
        meta[p] = {k: v for k, v in info.items() if not isinstance(v, np.ndarray)}
        for split in SPLITS:
            layer = "mainline_states" if split == "train" else "mainline_states_masked"
            V, Q, O, E = read_mainline(p, split, layer, info["links"])
            arrs[f"{split}_V"] = V.astype(np.float16); arrs[f"{split}_Q"] = Q.astype(np.float16)
            arrs[f"{split}_O"] = O.astype(np.float16); arrs[f"{split}_E"] = E
            if split == "train":
                Vm, _, _, _ = read_mainline(p, split, "mainline_states_masked", info["links"])
                arrs["train_Mavail"] = np.isfinite(Vm)          # what the masked layer releases
                del Vm
            R, RE = read_ramps(p, split, info["ramps"])
            arrs[f"{split}_R"] = R.astype(np.float16); arrs[f"{split}_RE"] = RE
            w = windows(p, split)
            w.to_csv(f"{OUT}/{p}_{split}_windows.csv", index=False)
            # official window history: check against the layer, then overlay any cell the layer lacks
            h = pd.read_parquet(f"{DATA}/task2/{p}/{split}/window_history.parquet")
            t0 = pd.Timestamp(SPLITS[split][0], tz="UTC")
            rr = ((pd.to_datetime(h.timestamp, utc=True) - t0) // STEP).to_numpy()
            cc = h.link_id.astype(str).map({l: i for i, l in enumerate(info["links"])}).to_numpy()
            hv = h.speed_kmh.to_numpy(np.float32); lv = V[rr, cc]
            both = np.isfinite(hv) & np.isfinite(lv)
            add = np.isfinite(hv) & ~np.isfinite(lv)
            print(f"  {p} {split} history check: cols {list(h.columns)} rows {len(h)} hist nonnull {np.isfinite(hv).mean():.3f} layer nonnull {np.isfinite(lv).mean():.3f} "
                  f"max|diff| {np.nanmax(np.abs(hv[both]-lv[both])) if both.any() else -1:.4f} hist-only {int(add.sum())} layer-only {int((~np.isfinite(hv)&np.isfinite(lv)).sum())}", flush=True)
            if add.any():
                V[rr[add], cc[add]] = hv[add]
                if "flow_vph" in h: Q[rr[add], cc[add]] = h.flow_vph.to_numpy(np.float32)[add]
                if "occupancy" in h: O[rr[add], cc[add]] = h.occupancy.to_numpy(np.float32)[add]
                if "is_score_eligible" in h: E[rr[add], cc[add]] = h.is_score_eligible.to_numpy()[add].astype(np.int8)
                arrs[f"{split}_V"] = V.astype(np.float16); arrs[f"{split}_Q"] = Q.astype(np.float16)
                arrs[f"{split}_O"] = O.astype(np.float16); arrs[f"{split}_E"] = E
        np.savez_compressed(f"{OUT}/{p}.npz", **arrs)
        print(p, "saved", os.path.getsize(f"{OUT}/{p}.npz") / 1e6, "MB", el(), flush=True)
    json.dump(meta, open(f"{OUT}/meta.json", "w"))
    print("DONE", el())


if __name__ == "__main__":
    _run_prep()
