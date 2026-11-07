"""Stage 06 - Task 4 path flows. For every corridor and split: min ||A f - c||^2 + lambda ||f - b||^2, f >= 0, lambda = 0.05, solved exactly by
non-negative least squares (scipy.optimize.nnls) on the split's own released link counts c and weak prior b, measured links only. This is the
regularized solve the organizers' build_task4_odme_artifacts.py describes; the public 0.86517 pipeline (anubisyy) reaches the same optimum
with L-BFGS-B (mean relative L1 difference 1.4e-6). Writes odme_rows.csv keyed by panel, departure_time, path_id."""
# ---- paths: Kaggle by default; run_all.py points these at local folders -------------------------------------
import os as _os, glob as _glob
TFB_ROOTS = [r for r in _os.environ.get("TFB_INPUT", "/kaggle/input").split(_os.pathsep) if r]
TFB_OUT = _os.environ.get("TFB_OUT", "/kaggle/working"); _os.makedirs(TFB_OUT, exist_ok=True)
def G(pat):
    """every file matching pat under the input roots (recursive), sorted"""
    return sorted(set(p for r in TFB_ROOTS for p in _glob.glob(_os.path.join(r, "**", pat), recursive=True)))
# ----------------------------------------------------------------------------------------------------------------
import time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import nnls
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
release = Path(G('kaggle_public')[0])
def released_counts(panel, split):
    return pd.read_csv(release / "task4" / panel / split / "synthetic_link_counts.csv", dtype={"link_id": str})[["link_id", "count"]]
def released_prior(panel, split):
    return pd.read_csv(release / "task4" / panel / split / "synthetic_weak_prior.csv", dtype={"path_id": str})
def load_operator(network):
    paths = pd.read_csv(network / "path_set.csv"); paths["path_id"] = paths.path_id.astype(str)
    inc = pd.read_csv(network / "path_link_incidence.csv"); inc["path_id"] = inc.path_id.astype(str); inc["link_id"] = inc.link_id.astype(str)
    path_ids = paths.path_id.tolist(); link_ids = inc.link_id.drop_duplicates().tolist()
    pi = {x: i for i, x in enumerate(path_ids)}; li = {x: i for i, x in enumerate(link_ids)}
    A = np.zeros((len(link_ids), len(path_ids))); A[inc.link_id.map(li).to_numpy(), inc.path_id.map(pi).to_numpy()] = 1.0
    return path_ids, link_ids, A
LAM = 0.05; out = []
for panel in sorted(p.name for p in (release / 'corridors').iterdir() if p.is_dir()):
    path_ids, link_ids, A = load_operator(release / 'corridors' / panel / 'network')
    for split in ('validation', 'private'):
        cf = released_counts(panel, split); pr = released_prior(panel, split)
        c = cf.set_index('link_id').reindex(link_ids)['count'].to_numpy(float); m = np.isfinite(c)
        b = pd.to_numeric(pr.set_index(pr.path_id.astype(str)).reindex(path_ids).path_flow, errors='coerce').fillna(0.0).to_numpy(float)
        Am = A[m]
        f, _ = nnls(np.vstack([Am, np.sqrt(LAM) * np.eye(Am.shape[1])]), np.concatenate([c[m], np.sqrt(LAM) * b]), maxiter=50 * Am.shape[1])
        out.append(pd.DataFrame({'panel': panel, 'departure_time': str(pr.departure_time.iloc[0]), 'path_id': path_ids, 'path_flow': f}))
        log(panel, split, 'paths', len(path_ids), 'measured links', int(m.sum()), 'S_link', round(1 - np.abs(Am @ f - c[m]).sum() / c[m].sum(), 6))
o = pd.concat(out, ignore_index=True); assert (o.path_flow >= 0).all() and np.isfinite(o.path_flow).all()
o.to_csv(_os.path.join(TFB_OUT, 'odme_rows.csv'), index=False); log('rows', len(o))
