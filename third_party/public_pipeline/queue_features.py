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
        sample = sorted((RELEASE / "corridors" / name / "train" / "mainline_states" /
                         "year_month=2030-06").glob("*.parquet"))[0]
        location = pd.read_parquet(sample, columns=["link_id", "milepost"]).drop_duplicates("link_id")
        location["link_id"] = location.link_id.astype(str)
        milepost = location.set_index("link_id").reindex(self.links).milepost.to_numpy(dtype=np.float32)
        if not np.isfinite(milepost).all():
            raise ValueError(f"Missing mileposts for {name}")
        self.milepost = milepost
        self.sorted_links = np.argsort(milepost)
        self.link_rank = np.empty(len(self.links), dtype=np.int32)
        self.link_rank[self.sorted_links] = np.arange(len(self.links))
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
        # Short lags preserve the shape of an approaching shock wave. A mean over
        # the whole history loses the timing of a rapid pre-onset slowdown.
        for lag in (1, 2, 3, 6, 9, 12):
            features.extend((speed[-lag], flow[-lag]))
        features.extend((
            min_finite(speed[-3:]), min_finite(speed[-6:]),
            avg(speed[-3:]) - avg(speed[-6:-3]),
            avg(flow[-3:]) - avg(flow[-6:-3]),
            self.neighboring(self.incoming, avg(speed[-3:])),
            self.neighboring(self.outgoing, avg(speed[-3:])),
            self.neighboring(self.incoming, avg(flow[-3:])),
            self.neighboring(self.outgoing, avg(flow[-3:])),
            self.neighboring(self.incoming, queue_history[-1].astype(np.float32)),
            self.neighboring(self.outgoing, queue_history[-1].astype(np.float32)),
        ))
        def milepost_neighbor(values, offset):
            ranks = self.link_rank + offset
            valid_neighbor = (ranks >= 0) & (ranks < len(self.links))
            links = self.sorted_links[np.clip(ranks, 0, len(self.links)-1)]
            return np.where(valid_neighbor, values[links], np.nan)

        features.append(self.milepost)
        for offset in (-2, -1, 1, 2):
            neighbor_milepost = milepost_neighbor(self.milepost, offset)
            features.extend((
                np.abs(neighbor_milepost-self.milepost),
                milepost_neighbor(last_speed, offset),
                milepost_neighbor(recent_speed, offset),
                milepost_neighbor(last_flow, offset),
                milepost_neighbor(recent_flow, offset),
                milepost_neighbor(last_queue.astype(np.float32), offset),
            ))
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
