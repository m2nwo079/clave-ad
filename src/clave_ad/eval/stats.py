"""AUROC and category-stratified image bootstrap (design doc 12, 13.1 A8), numpy and scipy only."""

from typing import Callable

import numpy as np
from scipy.stats import rankdata


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Area under the ROC curve with average ranks (ties count 0.5); None when one class is missing."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def resample_indices(strata: np.ndarray, resamples: int, seed: int) -> list[np.ndarray]:
    """Image indices of each resample: every stratum is drawn with replacement at its own size."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    return [np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups]) for _ in range(resamples)]


def interval(values: list[float], level: float) -> list[float] | None:
    """Percentile interval of the kept resample values."""
    if not values:
        return None
    tail = 100 * (1 - level) / 2
    return [float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))]


def bootstrap(statistic: Callable[[np.ndarray], float | None], indices: list[np.ndarray], level: float) -> dict:
    """Interval of one statistic over shared resamples; a resample where it is undefined is dropped and counted."""
    values = [v for v in (statistic(i) for i in indices) if v is not None]
    return {"ci": interval(values, level), "dropped": len(indices) - len(values)}


def _self_check() -> None:
    # perfect separation, reversed, and all ties
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5
    # hand count: positives 0.5, 0.9; negatives 0.5, 0.1 -> pairs (0.5 vs 0.5) 0.5, three wins -> 3.5 / 4
    assert auroc(np.array([0.5, 0.9, 0.5, 0.1]), np.array([1, 1, 0, 0])) == 0.875
    assert auroc(np.array([0.3, 0.4]), np.array([1, 1])) is None
    # same as the pairwise definition on random data with many ties
    rng = np.random.default_rng(1)
    s, y = rng.integers(0, 4, 200) / 3, rng.random(200) < 0.4
    pairs = (s[y][:, None] > s[~y][None, :]) + 0.5 * (s[y][:, None] == s[~y][None, :])
    assert abs(auroc(s, y) - pairs.mean()) < 1e-12

    strata = np.array(["a"] * 5 + ["b"] * 3)
    first, again = resample_indices(strata, 50, 0), resample_indices(strata, 50, 0)
    assert all((x == z).all() for x, z in zip(first, again))
    # stratum sizes are kept in every resample
    assert all((strata[i] == "a").sum() == 5 and (strata[i] == "b").sum() == 3 for i in first)
    # a stratum holding the only positive makes some resamples single-class; those are dropped and counted
    labels = np.array([1, 0, 0, 0, 0, 0, 0, 0], bool)
    scores = np.arange(8.0)
    out = bootstrap(lambda i: auroc(scores[i], labels[i]), first, 0.95)
    assert out["dropped"] == sum(not labels[i].any() for i in first) > 0
    assert out["ci"][0] <= out["ci"][1]
    print("stats self-check ok")


if __name__ == "__main__":
    _self_check()
