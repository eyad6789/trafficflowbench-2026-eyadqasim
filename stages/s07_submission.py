"""Stage 07 - the selected submission (public 0.88236, private 0.87520, file sub_e50.csv). Joins the outputs of stages 03-06 on the
keys of submission_key.csv, in its order:
  state  : speed_kmh, flow_vph from stage 03 (mean of the U-Net seeds, density step w = 0.5)
  queue  : onset rows    - only the T+30 step can be queued (the official onset windows are the first feasible origins, so the first queue
                           sits at T+30); p = mean(LightGBM bag, CatBoost; with and without link identity), for the first window of each
                           corridor and month p = 0.6 p + 0.4 p_public; queued where p >= 0.55, the most likely link if none is.
           ongoing rows  - p = mean of the two stage-05 models; queued where p >= 0.5; a window with no queued cell gets the
                           expected-IoU set.
  odme   : path_flow from stage 06.
If a copy of the submitted file (sub_e50.csv) is among the inputs, the result is compared with it cell by cell."""
# ---- paths: Kaggle by default; run_all.py points these at local folders -------------------------------------
import os as _os, glob as _glob
TFB_ROOTS = [r for r in _os.environ.get("TFB_INPUT", "/kaggle/input").split(_os.pathsep) if r]
TFB_OUT = _os.environ.get("TFB_OUT", "/kaggle/working"); _os.makedirs(TFB_OUT, exist_ok=True)
def G(pat):
    """every file matching pat under the input roots (recursive), sorted"""
    return sorted(set(p for r in TFB_ROOTS for p in _glob.glob(_os.path.join(r, "**", pat), recursive=True)))
# ----------------------------------------------------------------------------------------------------------------
import time
import numpy as np, pandas as pd
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
def one(pat):
    f = G(pat); assert len(f) == 1, (pat, f); log(pat, '<-', f[0]); return f[0]
DATA = _os.path.dirname(one('kaggle_public/submission_key.csv')); PREP = _os.path.dirname(one('prep/meta.json'))
key = pd.read_csv(f'{DATA}/submission_key.csv', dtype=str)
log('submission_key', len(key), key.task.value_counts().to_dict())
sub = key[['submission_id', 'task']].copy()
for c in ('speed_kmh', 'flow_vph', 'queue_pred', 'path_flow'): sub[c] = 0.0
TASK2_ONLY = _os.environ.get('TFB_TASK2_ONLY', '0') == '1'
if TASK2_ONLY:
    log('Task 2 only: the state and odme columns stay 0')
else:
    # state
    st = pd.read_csv(one('state_rows.csv')); st['submission_id'] = st.submission_id.astype(str); st = st.set_index('submission_id')
    ms = (sub.task == 'state').to_numpy()
    sub.loc[ms, ['speed_kmh', 'flow_vph']] = st.reindex(sub.submission_id[ms])[['speed_kmh', 'flow_vph']].to_numpy()
    # odme
    od = pd.read_csv(one('odme_rows.csv'), dtype={'panel': str, 'departure_time': str, 'path_id': str}).set_index(['panel', 'departure_time', 'path_id'])
    mo = (sub.task == 'odme').to_numpy()
    sub.loc[mo, 'path_flow'] = od.reindex(pd.MultiIndex.from_frame(key.loc[mo, ['panel', 'departure_time', 'path_id']])).path_flow.to_numpy()
# queue
mq = (sub.task == 'queue').to_numpy(); kq = key.loc[mq, ['submission_id', 'window_id', 'timestamp', 'link_id']].reset_index(drop=True)
wins = pd.concat([pd.read_csv(f) for f in G('prep/*_windows.csv') if not f.endswith('_train_windows.csv')], ignore_index=True)
cond = kq.window_id.map(wins.drop_duplicates('window_id').set_index('window_id').condition); assert cond.notna().all()
onset = (cond == 'queue_onset').to_numpy(); wid = kq.window_id.to_numpy()
on = pd.read_csv(one('onset_probs.csv'), dtype={'submission_id': str}).set_index('submission_id')
og = pd.read_csv(one('ongoing_probs.csv'), dtype={'submission_id': str}).set_index('submission_id')
ids = kq.submission_id.to_numpy()
last = onset & on.reindex(ids).Bag_none.notna().to_numpy()
assert last.sum() == len(on), (int(last.sum()), len(on))
DD = on[['Bag_none', 'Bag_identity', 'Cat_none', 'Cat_identity']].reindex(ids).mean(axis=1).fillna(0).to_numpy()
kk = on.k.reindex(ids).fillna(9).to_numpy(); pub = on.pub.reindex(ids).fillna(0).to_numpy()
w = np.where(kk == 0, .6, 1.0); PN = w * DD + (1 - w) * pub
EE = (.5 * og.p_e + .5 * og.p_n).reindex(ids).to_numpy(); assert np.isfinite(EE[~onset]).all()
def eiou_k(ps):
    o = np.argsort(-ps); s = ps[o]; cs = np.cumsum(s); k = np.arange(1, len(s) + 1)
    return o[:int(np.argmax(cs / (k + (s.sum() - cs)))) + 1]
v = np.zeros(len(kq), int)
for w_, g in pd.DataFrame({'w': wid[last], 'i': np.flatnonzero(last)}).groupby('w'):
    ii = g.i.to_numpy(); s_ = PN[ii] >= .55
    if not s_.any(): s_[np.argmax(PN[ii])] = True
    v[ii[s_]] = 1
io = np.flatnonzero(~onset); v[io] = (EE[io] >= .5).astype(int)
for w_, g in pd.DataFrame({'w': wid[io], 'v': v[io], 'i': io}).groupby('w'):
    if g.v.sum() == 0:
        ii = g.i.to_numpy(); v[ii[eiou_k(EE[ii])]] = 1
sub.loc[mq, 'queue_pred'] = v
sub['queue_pred'] = sub.queue_pred.astype(int)
assert np.isfinite(sub[['speed_kmh', 'flow_vph', 'path_flow']].to_numpy()).all() and (sub.path_flow >= 0).all() and set(sub.queue_pred.unique()) <= {0, 1}
assert len(sub) == len(key) and sub.submission_id.is_unique
log('rows', len(sub), '(the released submission_key.csv has 6,980,503)')
out = _os.path.join(TFB_OUT, 'submission.csv'); sub.to_csv(out, index=False)
log('written', out, 'onset cells', int(v[onset].sum()), 'ongoing cells', int(v[~onset].sum()))
ref = G('sub_e50.csv')
if ref:
    r = pd.read_csv(ref[0]); r['submission_id'] = r.submission_id.astype(str); r = r.set_index('submission_id').reindex(sub.submission_id.to_numpy())
    if not r.task.notna().all():
        log(ref[0], 'has other submission ids (another month?); comparison skipped')
    else:
        for t, cols in (() if TASK2_ONLY else (('state', ['speed_kmh', 'flow_vph']), ('odme', ['path_flow']))):
            m = (sub.task == t).to_numpy()
            for c in cols:
                a = sub[c].to_numpy()[m]; b = r[c].to_numpy()[m]
                log(f'vs submitted {t}.{c}: mean |diff| {np.mean(np.abs(a - b)):.5f}, max |diff| {np.max(np.abs(a - b)):.4f}, mean value {np.mean(np.abs(b)):.3f}')
        a = sub.queue_pred.to_numpy()[mq]; b = r.queue_pred.to_numpy()[mq].astype(int)
        log(f'vs submitted queue: onset cells {int(a[onset].sum())} / {int(b[onset].sum())} shared {int((a[onset] & b[onset]).sum())}; ongoing cells {int(a[~onset].sum())} / {int(b[~onset].sum())} shared {int((a[~onset] & b[~onset]).sum())}')
