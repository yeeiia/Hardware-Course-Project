from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def make_partitions(labels: list[int], num_clients: int, partition_config: dict[str, Any], seed: int) -> list[list[int]]:
    kind = str(partition_config.get("type", "iid")).lower()
    if kind == "iid":
        partitions = _iid(labels, num_clients, seed)
    elif kind == "dirichlet":
        alpha = float(partition_config.get("dirichlet_alpha", 0.3))
        partitions = _dirichlet(labels, num_clients, alpha, seed)
    elif kind == "quantity_skew":
        ratios = partition_config.get("quantity_ratios", [0.7, 0.3])
        partitions = _quantity_skew(labels, num_clients, ratios, seed)
    else:
        raise ValueError(f"unknown partition type: {kind}")
    for client_id, indices in enumerate(partitions):
        if not indices:
            raise ValueError(f"client {client_id} received no samples")
    return partitions


def label_histogram(labels: list[int], indices: list[int]) -> dict[int, int]:
    return dict(sorted(Counter(labels[i] for i in indices).items()))


def _iid(labels: list[int], num_clients: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[] for _ in range(num_clients)]
    labels_array = np.array(labels)
    for label in sorted(set(labels)):
        indices = np.where(labels_array == label)[0]
        rng.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        for cid, split in enumerate(splits):
            partitions[cid].extend(int(i) for i in split)
    for part in partitions:
        rng.shuffle(part)
    return partitions


def _dirichlet(labels: list[int], num_clients: int, alpha: float, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    labels_array = np.array(labels)
    partitions: list[list[int]] = [[] for _ in range(num_clients)]
    for label in sorted(set(labels)):
        indices = np.where(labels_array == label)[0]
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        split_points = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
        splits = np.split(indices, split_points)
        for cid, split in enumerate(splits):
            partitions[cid].extend(int(i) for i in split)
    _repair_empty_partitions(partitions, rng)
    for part in partitions:
        rng.shuffle(part)
    return partitions


def _quantity_skew(labels: list[int], num_clients: int, ratios: list[float], seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    ratios_array = np.array(ratios[:num_clients], dtype=float)
    if len(ratios_array) < num_clients:
        ratios_array = np.pad(ratios_array, (0, num_clients - len(ratios_array)), constant_values=1.0)
    ratios_array = ratios_array / ratios_array.sum()
    split_points = (np.cumsum(ratios_array)[:-1] * len(indices)).astype(int)
    return [[int(i) for i in split] for split in np.split(indices, split_points)]


def _repair_empty_partitions(partitions: list[list[int]], rng: np.random.Generator) -> None:
    for cid, part in enumerate(partitions):
        if part:
            continue
        donor = max(range(len(partitions)), key=lambda i: len(partitions[i]))
        move_pos = int(rng.integers(0, len(partitions[donor])))
        part.append(partitions[donor].pop(move_pos))
