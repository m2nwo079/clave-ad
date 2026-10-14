"""VLM inputs derived from the anomaly map: the B2 colour map beside the original, and the zoom-in re-query pair
(half-size original, crop at original resolution). Each input image comes with a hash for cache keys and records.
"""

import hashlib

import numpy as np
from PIL import Image

from clave_ad.data.maps import downscale, jet_image, zoom_box


def image_sha256(image: Image.Image) -> str:
    """Hash of a derived input image (B2 map, half-size original, crop), for the cache key and the record."""
    return hashlib.sha256(f"{image.mode}{image.size}".encode() + image.tobytes()).hexdigest()


def b2_inputs(original: Image.Image, original_sha: str, anomaly_map: np.ndarray,
              scale: dict) -> tuple[list[tuple[Image.Image, str]], dict]:
    """Original and its JET map on the fixed dataset scale, and the record fields that describe them."""
    jet = jet_image(anomaly_map, original.size, scale["low"], scale["high"])
    inputs = [(original, original_sha), (jet, image_sha256(jet))]
    return inputs, {"inputs_sha256": [h for _, h in inputs]}


def requery_inputs(original: Image.Image, anomaly_map: np.ndarray,
                   requery: dict) -> tuple[list[tuple[Image.Image, str]], dict]:
    """Half-size original and the zoom-in crop (A3), and the record fields that describe them."""
    box, shift = zoom_box(anomaly_map, original.size, requery["top_fraction"], requery["box_scale"], requery["grid"])
    half, crop = downscale(original, requery["downscale"]), original.crop(box)
    inputs = [(half, image_sha256(half)), (crop, image_sha256(crop))]
    return inputs, {"inputs_sha256": [h for _, h in inputs], "crop_box": list(box), "crop_size": list(crop.size),
                    "crop_shift": list(shift)}
