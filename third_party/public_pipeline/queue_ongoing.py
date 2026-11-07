"""Train a causal queue classifier on observed train months and predict Task 2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse

from config import RELEASE, OUT, THREADS
MONTHS = {"2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01", "2031-02"}
HISTORY = 12
HORIZON = 6
SEED = 20260930


def panel_names():
    return [entry["corridor_id"] for entry in json.loads((RELEASE / "config" / "corridors.json").read_text())["panels"]
            if entry["corridor_id"] not in {"D12_I405_N", "D12_I405_S"}]


def avg(values):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    result = np.nansum(values, axis=0) / np.maximum(count, 1)
    return np.where(count > 0, result, np.nan)


def min_finite(values):
    result = np.where(np.isfinite(values), values, np.inf).min(axis=0)
    return np.where(np.isfinite(result), result, np.nan)


@dataclass
class Window:
    X: np.ndarray
    y: np.ndarray
    eligible: np.ndarray
    panel: str
    condition: str


class Panel:
    def __init__(self, name, panel_no):
        self.name = name
        self.panel_no = panel_no
        directory = RELEASE / "corridors" / name / "network"
        fd = pd.read_csv(directory / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id")
        topo = pd.read_csv(directory / "lwr_mainline_topology.csv", dtype=str)
        self.links = fd.link_id.astype(str).tolist()
        self.link_index = {key: i for i, key in enumerate(self.links)}
        self.lanes = fd.lanes.to_numpy(dtype=np.float32)
        self.vf = fd.free_speed_kmh.to_numpy(dtype=np.float32)
        self.cap = fd.capacity_vph.to_numpy(dtype=np.float32)
        self.length = fd.length_km.to_numpy(dtype=np.float32)
        self.cut = (fd.v_cut if "v_cut" in fd else 0.6 * fd.free_speed_kmh).to_numpy(dtype=np.float32)
        self.incoming = self.adjacency(topo, "incoming_link_ids")
        self.outgoing = self.adjacency(topo, "outgoing_link_ids")

    def adjacency(self, topo, column):
        rows, cols = [], []
        for row in topo.itertuples(index=False):
            link = str(row.link_id)
            if link not in self.link_index:
                continue
            for other in str(getattr(row, column)).split(";"):
                if other in self.link_index:
                    rows.append(self.link_index[link])
                    cols.append(self.link_index[other])
        result = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                   shape=(len(self.links), len(self.links)))
        result.data[:] = 1.0
        return result

    def neighboring(self, graph, values):
        valid = np.isfinite(values).astype(np.float32)
        count = graph @ valid
        mean = graph @ np.nan_to_num(values, nan=0.0)
        return np.where(count > 0, mean / np.maximum(count, 1), np.nan)

    def features(self, speed, flow, observed, origin, condition):
        """History arrays have exactly twelve rows ending five minutes before T."""
        if speed.shape != (HISTORY, len(self.links)) or flow.shape != speed.shape:
            raise ValueError((self.name, speed.shape, flow.shape))
        speed = speed.astype(np.float32) / self.vf[None, :]
        flow = flow.astype(np.float32) / self.cap[None, :]
        speed = np.where(np.isfinite(speed), speed, np.nan)
        flow = np.where(np.isfinite(flow), flow, np.nan)
        valid = np.isfinite(speed)
        positions = np.where(valid, np.arange(HISTORY)[:, None], -1).max(axis=0)
        columns = np.arange(len(self.links))
        last_speed = np.where(positions >= 0, speed[np.maximum(positions, 0), columns], np.nan)
        last_flow = np.where(positions >= 0, flow[np.maximum(positions, 0), columns], np.nan)
        age = HISTORY - 1 - positions
        age = np.where(positions >= 0, age, 12)
        recent_speed = avg(speed[-3:])
        prior_speed = avg(speed[-6:-3])
        recent_flow = avg(flow[-3:])
        prior_flow = avg(flow[-6:-3])
        mean_speed = avg(speed)
        mean_flow = avg(flow)
        queue_history = np.where(valid, speed <= (self.cut / self.vf)[None, :], False)
        last_queue = last_speed <= self.cut / self.vf
        hour = origin.hour + origin.minute / 60
        tod_sin = np.sin(2 * np.pi * hour / 24)
        tod_cos = np.cos(2 * np.pi * hour / 24)
        dow = origin.dayofweek
        constants = lambda value: np.full(len(self.links), value, dtype=np.float32)
        features = [
            constants(self.panel_no), np.arange(len(self.links), dtype=np.float32),
            self.vf, self.cap / np.maximum(self.lanes, 1), self.lanes, self.length,
            constants(hour), constants(tod_sin), constants(tod_cos), constants(dow),
            constants(1 if condition == "queue_ongoing" else 0),
            last_speed, last_flow, constants(0) + age, recent_speed, prior_speed,
            recent_speed - prior_speed, recent_flow, prior_flow, recent_flow - prior_flow,
            mean_speed, min_finite(speed), mean_flow, np.nanmax(np.where(np.isfinite(flow), flow, -np.inf), axis=0),
            valid.mean(axis=0), queue_history.sum(axis=0).astype(np.float32), last_queue.astype(np.float32),
            self.neighboring(self.incoming, last_speed), self.neighboring(self.outgoing, last_speed),
            self.neighboring(self.incoming, last_flow), self.neighboring(self.outgoing, last_flow),
            self.neighboring(self.incoming, last_queue.astype(np.float32)),
            self.neighboring(self.outgoing, last_queue.astype(np.float32)),
            constants(np.isfinite(speed).mean()), constants(np.nanmin(speed) if np.isfinite(speed).any() else 1.0),
        ]
        features[25] = np.where(np.isfinite(features[25]), features[25], np.nan)
        features[24] = np.where(np.isfinite(features[24]), features[24], np.nan)
        base = np.column_stack(features).astype(np.float32)
        X = np.tile(base, (HORIZON, 1))
        X = np.column_stack([X, np.repeat(np.arange(1, HORIZON + 1), len(self.links))]).astype(np.float32)
        X[~np.isfinite(X)] = np.nan
        return X

    def train_day(self, path):
        columns = ["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]
        frame = pd.read_parquet(path, columns=columns)
        frame["link_id"] = frame.link_id.astype(str)
        if frame.duplicated(["timestamp", "link_id"]).any():
            frame = frame.groupby(["timestamp", "link_id"], as_index=False).mean(numeric_only=True)
        times = sorted(frame.timestamp.unique())
        index = pd.Index(times)
        speed = frame.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=self.links).to_numpy(dtype=np.float32)
        flow = frame.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=self.links).to_numpy(dtype=np.float32)
        eligible = frame.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=self.links).fillna(0).to_numpy(dtype=bool)
        return times, speed, flow, eligible


def candidate_origins(panel, times, speed, eligible):
    queued = (speed <= panel.cut[None, :]) & eligible & np.isfinite(speed)
    active = queued.any(axis=1)
    candidates = {"queue_onset": [], "queue_ongoing": []}
    for t in range(HISTORY, len(times) - HORIZON):
        if np.isfinite(speed[t-HISTORY:t]).mean() < .7:
            continue
        if not active[t+1:t+HORIZON+1].any():
            continue
        if not active[t-HISTORY:t].any() and not active[t]:
            candidates["queue_onset"].append(t)
        if active[t]:
            candidates["queue_ongoing"].append(t)
    return candidates


def choose_origins(candidates, panel, date):
    output = []
    for condition, values in candidates.items():
        seed = int.from_bytes(hashlib.blake2b(f"{SEED}|{panel}|{date}|{condition}".encode(), digest_size=8).digest(), "big")
        rng = np.random.default_rng(seed)
        chosen = []
        for t in rng.permutation(values):
            if all(abs(t - other) >= 72 for other in chosen):
                chosen.append(int(t))
            if len(chosen) >= 2:
                break
        output.extend((condition, t) for t in chosen)
    return output


def collect():
    train_X, train_y, train_w = [], [], []
    valid_windows = []
    feature_count = None
    for panel_no, name in enumerate(panel_names()):
        panel = Panel(name, panel_no)
        files = sorted((RELEASE / "corridors" / name / "train" / "mainline_states").glob("**/*.parquet"))
        files = [path for path in files if path.parent.name.removeprefix("year_month=") in MONTHS]
        print(f"data {name}: {len(files)} days", flush=True)
        panel_counts = {"train_onset": 0, "train_ongoing": 0, "valid_onset": 0, "valid_ongoing": 0}
        for path in files:
            times, speed, flow, eligible = panel.train_day(path)
            choices = choose_origins(candidate_origins(panel, times, speed, eligible), name, times[0][:10])
            is_valid_month = path.parent.name.endswith("2031-02")
            for condition, t in choices:
                origin = pd.Timestamp(times[t])
                X = panel.features(speed[t-HISTORY:t], flow[t-HISTORY:t], eligible[t-HISTORY:t], origin, condition)
                future_speed = speed[t+1:t+HORIZON+1]
                y = ((future_speed <= panel.cut[None, :]) & np.isfinite(future_speed)).reshape(-1)
                valid = (eligible[t+1:t+HORIZON+1] & np.isfinite(future_speed)).reshape(-1)
                if valid.sum() < .6 * len(valid):
                    continue
                if is_valid_month:
                    valid_windows.append(Window(X, y, valid, name, condition))
                    panel_counts[f"valid_{'onset' if condition == 'queue_onset' else 'ongoing'}"] += 1
                else:
                    train_X.append(X[valid])
                    train_y.append(y[valid].astype(np.uint8))
                    train_w.append(np.full(valid.sum(), 1000 / valid.sum(), dtype=np.float32))
                    panel_counts[f"train_{'onset' if condition == 'queue_onset' else 'ongoing'}"] += 1
                feature_count = X.shape[1]
        print(panel_counts, flush=True)
    X = np.vstack(train_X)
    y = np.concatenate(train_y)
    weights = np.concatenate(train_w)
    print(f"train: {X.shape}, positive={y.mean():.5f}; validation windows={len(valid_windows)}", flush=True)
    return X, y, weights, valid_windows, feature_count


def score_windows(windows, probabilities, threshold, onset_min):
    by_panel_condition = {}
    for window, p in zip(windows, probabilities):
        prediction = p >= threshold
        if window.condition == "queue_onset" and prediction.sum() < onset_min:
            index = np.argpartition(p, -onset_min)[-onset_min:]
            prediction[index] = True
        true = window.y & window.eligible
        predicted = prediction & window.eligible
        union = np.logical_or(true, predicted).sum()
        iou = np.logical_and(true, predicted).sum() / max(union, 1)
        by_panel_condition.setdefault((window.panel, window.condition), []).append(iou)
    panels = panel_names()
    condition_means = {key: float(np.mean(values)) for key, values in by_panel_condition.items()}
    overall = np.mean([np.mean([condition_means.get((panel, c), 0) for c in ("queue_onset", "queue_ongoing")]) for panel in panels])
    onset = np.mean([condition_means.get((panel, "queue_onset"), 0) for panel in panels])
    ongoing = np.mean([condition_means.get((panel, "queue_ongoing"), 0) for panel in panels])
    return float(overall), float(onset), float(ongoing)


def fit_model():
    X, y, weights, windows, feature_count = collect()
    persistence = [((w.X[:, 11] <= .60) & (w.X[:, 13] <= 3)).astype(np.float32) for w in windows]
    print("proxy persistence IoU, onset, ongoing:", score_windows(windows, persistence, .5, 0), flush=True)
    X_val = np.vstack([w.X[w.eligible] for w in windows])
    y_val = np.concatenate([w.y[w.eligible] for w in windows]).astype(np.uint8)
    w_val = np.concatenate([np.full(w.eligible.sum(), 1000 / w.eligible.sum(), dtype=np.float32) for w in windows])
    train_set = lgb.Dataset(X, label=y, weight=weights, free_raw_data=True)
    valid_set = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_set, free_raw_data=True)
    params = {
        "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.045,
        "num_leaves": 31, "max_depth": 8, "min_data_in_leaf": 250,
        "feature_fraction": .85, "bagging_fraction": .85, "bagging_freq": 1,
        "lambda_l2": 8.0, "max_bin": 127, "num_threads": THREADS,
        "verbosity": -1, "seed": SEED,
    }
    model = lgb.train(params, train_set, num_boost_round=700, valid_sets=[valid_set],
                      callbacks=[lgb.early_stopping(60, verbose=True), lgb.log_evaluation(50)])
    model.save_model(str(OUT / "queue_lgb.txt"))
    del X, X_val, train_set, valid_set
    probs = [model.predict(w.X, num_threads=THREADS) for w in windows]
    trials = []
    for threshold in np.arange(.05, .501, .025):
        for onset_min in (0, 3, 6, 12, 24):
            overall, onset, ongoing = score_windows(windows, probs, float(threshold), onset_min)
            trials.append((overall, onset, ongoing, float(threshold), onset_min))
    trials.sort(reverse=True)
    print("best threshold trials", trials[:10], flush=True)
    result = {"best_iteration": model.best_iteration, "feature_count": feature_count,
              "validation_windows": len(windows), "positive_fraction": float(y.mean()),
              "threshold": trials[0][3], "onset_min": trials[0][4],
              "proxy_iou": trials[0][0], "proxy_onset_iou": trials[0][1],
              "proxy_ongoing_iou": trials[0][2]}
    (OUT / "queue_model.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result, flush=True)
    return model, result
