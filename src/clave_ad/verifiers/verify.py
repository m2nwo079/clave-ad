"""Claim verification against the detector, without the VLM or ground truth (configs/main_run/rules.yaml).

Existence: the image score against the category threshold. Location: share of the top map pixels in the claimed
cells. Type: nearest prototype of the main-seed library to the crop embedding; a type with no prototype in the
category is unverifiable and leaves the score denominator.
"""

import numpy as np


def nearest_type(embedding: np.ndarray, prototypes: dict[str, np.ndarray], order: list[str]) -> str:
    """Type whose prototype has the highest cosine similarity; ties go to the earlier type in order."""
    best, best_sim = None, -np.inf
    for name in order:
        if name in prototypes:
            sim = float(embedding @ prototypes[name])
            if sim > best_sim:
                best, best_sim = name, sim
    return best


def verify(answer: dict, score: float, threshold: float, cell_shares: dict[str, float], embedding: np.ndarray,
           prototypes: dict[str, np.ndarray], order: list[str], min_share: float) -> dict:
    """Per-claim result (True, False, None when the claim is absent, "unverifiable") and the verification score."""
    claims = {"existence": answer["is_defective"] == (score > threshold), "location": None, "type": None}
    if answer["is_defective"]:
        cells = set(answer["location_cells"])
        claims["location"] = bool(cells) and sum(cell_shares[c] for c in cells) >= min_share
        if answer["defect_type"] in prototypes:
            claims["type"] = nearest_type(embedding, prototypes, order) == answer["defect_type"]
        else:
            claims["type"] = "unverifiable"
    checked = [v for v in claims.values() if isinstance(v, bool)]
    return {"claims": claims, "score": sum(checked) / len(checked), "all_pass": all(checked),
            "failed": not all(checked), "unverifiable": claims["type"] == "unverifiable"}


def _self_check() -> None:
    shares = {f"r{i}c{j}": 0.0 for i in (1, 2, 3) for j in (1, 2, 3)}
    shares.update({"r1c1": 0.6, "r1c2": 0.4})
    protos = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    order = ["a", "b", "c"]
    yes = {"is_defective": True, "defect_type": "a", "location_cells": ["r1c1"]}
    out = verify(yes, 0.9, 0.5, shares, np.array([1.0, 0.0]), protos, order, 0.5)
    assert out["claims"] == {"existence": True, "location": True, "type": True} and out["score"] == 1.0
    # location below half, wrong type
    out = verify({**yes, "defect_type": "b", "location_cells": ["r1c2"]}, 0.9, 0.5, shares, np.array([1.0, 0.0]),
                 protos, order, 0.5)
    assert out["claims"]["location"] is False and out["claims"]["type"] is False and out["failed"]
    # a type without prototype is unverifiable and leaves the denominator
    out = verify({**yes, "defect_type": "c"}, 0.9, 0.5, shares, np.array([1.0, 0.0]), protos, order, 0.5)
    assert out["claims"]["type"] == "unverifiable" and out["score"] == 1.0 and not out["failed"]
    # empty cells fail; a normal answer has the existence claim only, and score == threshold counts as normal
    assert verify({**yes, "location_cells": []}, 0.9, 0.5, shares, np.zeros(2), protos, order, 0.5)["claims"]["location"] is False
    out = verify({"is_defective": False}, 0.5, 0.5, shares, np.zeros(2), protos, order, 0.5)
    assert out["claims"] == {"existence": True, "location": None, "type": None} and out["score"] == 1.0
    # ties go to the earlier type
    assert nearest_type(np.array([1.0, 1.0]), protos, order) == "a"
    print("verify self-check ok")


if __name__ == "__main__":
    _self_check()
