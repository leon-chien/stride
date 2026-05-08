from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class ResampleStats:
    """
    Summary statistics from one weighted-ensemble resampling step.
    """

    num_input_walkers: int
    num_output_walkers: int
    num_occupied_bins: int
    total_weight_before: float
    total_weight_after: float
    min_bin_id: int
    max_bin_id: int


def group_indices_by_bin(bin_ids: np.ndarray) -> dict[int, list[int]]:
    """
    Group walker indices by integer bin ID.
    """
    groups: dict[int, list[int]] = defaultdict(list)

    for i, bin_id in enumerate(bin_ids):
        groups[int(bin_id)].append(i)

    return dict(groups)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize weights so they sum to one.
    """
    weights = np.asarray(weights, dtype=np.float64)

    total = weights.sum()

    if total <= 0:
        raise ValueError("Weights must have positive total mass.")

    return weights / total


def resample_bin_equal_weight(
    indices: list[int],
    weights: np.ndarray,
    target_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample one bin to target_count walkers.

    This is a simplified WESTPA-like split/merge operation.

    The total probability weight in the bin is conserved.
    Output walkers inside the bin receive equal weight:

        new_weight = total_bin_weight / target_count

    Parent walkers are selected according to their current weights.

    Args:
        indices:
            Indices of walkers currently in this bin.
        weights:
            Full population weights.
        target_count:
            Desired number of walkers in this bin.
        rng:
            NumPy random generator.

    Returns:
        parent_indices:
            Original walker indices selected as parents.
        new_weights:
            New weights for each selected child walker.
    """
    if target_count <= 0:
        raise ValueError("target_count must be positive.")

    if len(indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float64)

    indices_array = np.array(indices, dtype=int)

    bin_weights = weights[indices_array].astype(np.float64)
    total_bin_weight = float(bin_weights.sum())

    if total_bin_weight <= 0:
        raise ValueError("Bin has non-positive total weight.")

    parent_probs = bin_weights / total_bin_weight

    parent_indices = rng.choice(
        indices_array,
        size=target_count,
        replace=True,
        p=parent_probs,
    )

    new_weight = total_bin_weight / target_count
    new_weights = np.full(target_count, new_weight, dtype=np.float64)

    return parent_indices, new_weights


def allocate_counts_to_bins(
    bin_ids: list[int],
    target_total_count: int,
    bin_priorities: dict[int, float] | None = None,
    min_count_per_bin: int = 1,
) -> dict[int, int]:
    """
    Allocate output walker counts across occupied bins.

    If bin_priorities is None, walkers are distributed approximately
    uniformly across occupied bins.

    If bin_priorities is provided, bins with higher priority receive
    more walkers, while every occupied bin still receives at least
    min_count_per_bin walkers.

    This implements the Version 1.2 idea:

        diversity floor + priority-aware allocation

    Args:
        bin_ids:
            Occupied bin IDs.
        target_total_count:
            Total number of walkers desired after resampling.
        bin_priorities:
            Optional map from bin_id to nonnegative priority.
        min_count_per_bin:
            Minimum number of walkers assigned to each occupied bin.

    Returns:
        Dictionary mapping bin_id -> target walker count.
    """
    if target_total_count <= 0:
        raise ValueError("target_total_count must be positive.")

    if min_count_per_bin <= 0:
        raise ValueError("min_count_per_bin must be positive.")

    sorted_bins = sorted(int(b) for b in bin_ids)
    num_bins = len(sorted_bins)

    if num_bins == 0:
        raise ValueError("No bins provided.")

    if num_bins * min_count_per_bin > target_total_count:
        raise ValueError(
            "min_count_per_bin is too large for target_total_count. "
            f"{num_bins} bins × {min_count_per_bin} minimum walkers "
            f"> {target_total_count} target walkers."
        )

    counts = {bin_id: min_count_per_bin for bin_id in sorted_bins}
    remaining = target_total_count - num_bins * min_count_per_bin

    if remaining == 0:
        return counts

    if bin_priorities is None:
        priorities = np.ones(num_bins, dtype=np.float64)
    else:
        priorities = np.array(
            [max(float(bin_priorities.get(bin_id, 0.0)), 0.0) for bin_id in sorted_bins],
            dtype=np.float64,
        )

        if priorities.sum() <= 0:
            priorities = np.ones(num_bins, dtype=np.float64)

    priorities = priorities / priorities.sum()

    raw_extra = priorities * remaining
    extra_floor = np.floor(raw_extra).astype(int)
    leftover = int(remaining - extra_floor.sum())

    for bin_id, extra in zip(sorted_bins, extra_floor):
        counts[bin_id] += int(extra)

    if leftover > 0:
        fractional = raw_extra - extra_floor
        order = np.argsort(fractional)[::-1]

        for idx in order[:leftover]:
            counts[sorted_bins[int(idx)]] += 1

    assert sum(counts.values()) == target_total_count

    return counts


def weighted_ensemble_resample(
    walkers: list,
    weights: np.ndarray,
    bin_ids: np.ndarray,
    target_per_bin: int,
    rng: np.random.Generator,
    target_total_count: int | None = None,
    bin_priorities: dict[int, float] | None = None,
    min_count_per_bin: int = 1,
) -> tuple[list, np.ndarray, ResampleStats]:
    """
    Perform simplified weighted-ensemble resampling.

    Walkers are grouped by bin. Each occupied bin is resampled while
    conserving total probability weight inside that bin.

    Version 1.2 supports priority-aware allocation:

        - each occupied bin gets at least min_count_per_bin walkers
        - remaining walkers are allocated according to bin_priorities
        - if no priorities are provided, allocation is uniform

    This lets STRIDE preserve diversity while giving more walkers to
    high-value bins.
    """
    if len(walkers) != len(weights) or len(walkers) != len(bin_ids):
        raise ValueError(
            "walkers, weights, and bin_ids must have the same length. "
            f"Got {len(walkers)}, {len(weights)}, {len(bin_ids)}."
        )

    if target_per_bin <= 0:
        raise ValueError("target_per_bin must be positive.")

    weights = normalize_weights(weights)
    total_weight_before = float(weights.sum())

    groups = group_indices_by_bin(bin_ids)
    sorted_bins = sorted(groups.keys())
    num_occupied_bins = len(sorted_bins)

    if num_occupied_bins == 0:
        raise RuntimeError("No occupied bins found.")

    if target_total_count is not None:
        target_counts_by_bin = allocate_counts_to_bins(
            bin_ids=sorted_bins,
            target_total_count=target_total_count,
            bin_priorities=bin_priorities,
            min_count_per_bin=min_count_per_bin,
        )
    else:
        target_counts_by_bin = {
            bin_id: target_per_bin for bin_id in sorted_bins
        }

    new_walkers: list = []
    new_weights_list: list[float] = []

    for bin_id in sorted_bins:
        indices = groups[bin_id]
        target_count = target_counts_by_bin[bin_id]

        parent_indices, child_weights = resample_bin_equal_weight(
            indices=indices,
            weights=weights,
            target_count=target_count,
            rng=rng,
        )

        for parent_idx, child_weight in zip(parent_indices, child_weights):
            new_walkers.append(walkers[int(parent_idx)].clone())
            new_weights_list.append(float(child_weight))

    new_weights = np.array(new_weights_list, dtype=np.float64)
    new_weights = normalize_weights(new_weights)

    stats = ResampleStats(
        num_input_walkers=len(walkers),
        num_output_walkers=len(new_walkers),
        num_occupied_bins=len(groups),
        total_weight_before=total_weight_before,
        total_weight_after=float(new_weights.sum()),
        min_bin_id=int(np.min(bin_ids)),
        max_bin_id=int(np.max(bin_ids)),
    )

    return new_walkers, new_weights, stats


if __name__ == "__main__":
    class DummyWalker:
        def __init__(self, name: str) -> None:
            self.name = name

        def clone(self) -> "DummyWalker":
            return DummyWalker(self.name)

        def __repr__(self) -> str:
            return f"DummyWalker({self.name})"

    rng = np.random.default_rng(42)

    walkers = [DummyWalker(str(i)) for i in range(8)]
    weights = np.array([0.1, 0.2, 0.05, 0.15, 0.1, 0.1, 0.2, 0.1])
    bin_ids = np.array([0, 0, 0, 1, 1, 2, 2, 2])

    new_walkers, new_weights, stats = weighted_ensemble_resample(
        walkers=walkers,
        weights=weights,
        bin_ids=bin_ids,
        target_per_bin=2,
        rng=rng,
    )

    print(new_walkers)
    print(new_weights)
    print(stats)

    assert np.isclose(new_weights.sum(), 1.0)

    print("weighted_resampler sanity check passed.")