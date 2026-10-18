"""B4: detector-only template explanation without the VLM (design doc 11, 13.1 A6; configs/analysis/eval.yaml b4).

Existence uses the verification threshold, location takes cells by descending share of the top map pixels until the
cumulative share is reached, and type is the nearest library prototype to the A4 crop embedding.
"""

import numpy as np

from clave_ad.verifiers.verify import nearest_type


def b4_answer(score: float, threshold: float, cell_shares: dict[str, float], embedding: np.ndarray,
              prototypes: dict[str, np.ndarray], order: list[str], cumulative_share: float) -> dict:
    """Answer in the B0 format; a normal answer has the existence claim only."""
    if not score > threshold:
        return {"is_defective": False}
    # a stable sort keeps the row-major order of cell_shares among equal shares
    ranked = sorted(cell_shares, key=lambda c: -cell_shares[c])
    cells, total = [], 0.0
    for cell in ranked:
        cells.append(cell)
        total += cell_shares[cell]
        # tolerance for float sums of pixel shares that are exactly the target
        if total >= cumulative_share - 1e-9:
            break
    # a type without library images is never the nearest prototype, so B4 cannot answer it
    return {"is_defective": True, "defect_type": nearest_type(embedding, prototypes, order), "location_cells": cells}


def _self_check() -> None:
    shares = {f"r{i}c{j}": 0.0 for i in (1, 2, 3) for j in (1, 2, 3)}
    shares.update({"r2c2": 0.4, "r1c3": 0.2, "r3c1": 0.2, "r1c1": 0.2})
    protos = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    out = b4_answer(0.9, 0.5, shares, np.array([0.0, 1.0]), protos, ["a", "b", "c"], 0.5)
    # 0.4 alone is short of 0.5; among equal 0.2 shares r1c1 comes first in row-major order
    assert out == {"is_defective": True, "defect_type": "b", "location_cells": ["r2c2", "r1c1"]}
    # the cell reaching the share exactly is included and ends the list
    assert b4_answer(0.9, 0.5, shares, np.zeros(2) + 1, protos, ["a", "b"], 0.4)["location_cells"] == ["r2c2"]
    # score equal to the threshold is normal, as in the verifier
    assert b4_answer(0.5, 0.5, shares, np.zeros(2), protos, ["a"], 0.5) == {"is_defective": False}
    # "c" has no prototype, so it is never answered
    assert b4_answer(0.9, 0.5, shares, np.array([1.0, 1.0]), protos, ["c", "a", "b"], 0.5)["defect_type"] == "a"
    print("b4 self-check ok")


if __name__ == "__main__":
    _self_check()
