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
MONTHS = {"2030-06", "2030-11", "2031-01", "2031-02"}
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
