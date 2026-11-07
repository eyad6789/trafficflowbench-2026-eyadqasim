"""State rows from the no-memory U-Nets (tfb-t1-u3*: month-own profiles, trained on the train split + the released cells of March and April).
Mean over seeds, then the density head is folded in (w = 0.5 of the log-density move goes to flow), the setting that scored best on the held-out released cells."""
# ---- paths: Kaggle by default; run_all.py points these at local folders -------------------------------------
import os as _os, glob as _glob
TFB_ROOTS = [r for r in _os.environ.get("TFB_INPUT", "/kaggle/input").split(_os.pathsep) if r]
TFB_OUT = _os.environ.get("TFB_OUT", "/kaggle/working"); _os.makedirs(TFB_OUT, exist_ok=True)
def G(pat):
    """every file matching pat under the input roots (recursive), sorted"""
    return sorted(set(p for r in TFB_ROOTS for p in _glob.glob(_os.path.join(r, "**", pat), recursive=True)))
# ----------------------------------------------------------------------------------------------------------------
import glob, os
import numpy as np, pandas as pd
R = G('kaggle_public')[0]
fs = G('unet3_preds.parquet'); print('models', fs, flush=True)
U = None
for f in fs:
    d = pd.read_parquet(f, columns=['key', 'uv', 'uq', 'uk']).set_index('key')
    if U is None:
        U = d.copy(); U['uk'] = np.log(np.maximum(U.uk, .05))
    else:
        d = d.reindex(U.index); U['uv'] += d.uv; U['uq'] += d.uq; U['uk'] += np.log(np.maximum(d.uk, .05))
n = len(fs); v = np.clip(U.uv.to_numpy() / n, 3, 140); q = np.clip(U.uq.to_numpy() / n, 0, None); lk = U.uk.to_numpy() / n
W = 0.5
dd = np.clip(W * (lk - np.log(np.maximum(q, 1.0) / np.maximum(v, 1.0))), -1.0, 1.0)
q2 = np.clip(q * np.exp(dd), 0, None)
sk = pd.Index(U.index); assert sk.is_unique
ids, pos = [], []
for ch in pd.read_csv(f'{R}/submission_key.csv', dtype=str, chunksize=1_000_000, usecols=['submission_id', 'task', 'split', 'panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']):
    ch = ch[ch.task == 'state']
    kk = ch.split + '|' + ch.panel + '|' + ch.timestamp + '|' + ch.station_id + '|' + ch.link_id + '|' + ch.mask_regime
    ids.append(ch.submission_id.astype(np.int64).to_numpy()); pos.append(sk.get_indexer(kk))
ids = np.concatenate(ids); pos = np.concatenate(pos); assert (pos >= 0).all() and len(np.unique(pos)) == len(sk), (int((pos < 0).sum()), len(sk))
out = pd.DataFrame({'submission_id': ids, 'speed_kmh': v[pos], 'flow_vph': q2[pos]}).sort_values('submission_id')
assert np.isfinite(out[['speed_kmh', 'flow_vph']].to_numpy()).all()
out.to_csv(_os.path.join(TFB_OUT, 'state_rows.csv'), index=False, float_format='%.4f')
print('rows', len(out), 'speed', out.speed_kmh.min(), out.speed_kmh.max(), 'flow', out.flow_vph.min(), out.flow_vph.max(), 'low flow share', float((out.flow_vph < 50).mean()), 'mean |dq|/q', float(np.mean(np.abs(q2 - q) / np.maximum(q, 1))), flush=True)
