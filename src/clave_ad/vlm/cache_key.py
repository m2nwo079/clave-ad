"""Cache key of one VLM request in the main run, shared by the run and the evaluation."""

import hashlib
import json


def request_key(revision: str, generation: dict, kind: str, images: list[str], prompt: str, retry: str) -> str:
    # everything that changes the request goes into the key, as in the format check
    spec = {"revision": revision, "generation": generation, "kind": kind, "images": images, "prompt": prompt,
            "retry": retry, "forced": None}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
