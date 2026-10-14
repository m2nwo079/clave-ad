"""Anomaly map utilities: restoring the original size, the fixed-scale JET image (B2) and the zoom-in crop (A3).

Maps come at the detector resolution, made from the original image resized to a square without cropping.
"""

from math import ceil, floor

import numpy as np
from PIL import Image
from scipy import ndimage


def _jet() -> np.ndarray:
    # classic JET: dark blue at 0, cyan, yellow, dark red at 1
    t = np.linspace(0, 1, 256)[:, None]
    return (np.clip(1.5 - np.abs(4 * t - np.array([3, 2, 1])), 0, 1) * 255).round().astype(np.uint8)


JET = _jet()


def restore(anomaly_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The map resized back to the original (width, height) with bilinear interpolation."""
    return np.asarray(Image.fromarray(anomaly_map.astype(np.float32)).resize(size, Image.Resampling.BILINEAR))


def jet_image(anomaly_map: np.ndarray, size: tuple[int, int], low: float, high: float) -> Image.Image:
    """JET image of the restored map on a fixed scale; values outside [low, high] get the end colours."""
    t = np.clip((restore(anomaly_map, size) - low) / (high - low), 0, 1)
    return Image.fromarray(JET[np.round(t * 255).astype(np.uint8)])


def zoom_box(anomaly_map: np.ndarray, size: tuple[int, int], top_fraction: float, box_scale: float,
             grid: int) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Zoom-in crop (left, top, right, bottom) in original pixels, and how far it was moved (dx, dy) to fit the image.

    At map resolution: the 8-connected component of the top pixels that holds the peak. Its bounding box is moved to
    original coordinates, scaled about its centre and widened to at least one grid cell. A box crossing the image edge
    is moved inside with its size kept; an axis longer than the image is cut to the whole image (shift 0).
    """
    top = anomaly_map >= np.percentile(anomaly_map, 100 * (1 - top_fraction))
    labels, _ = ndimage.label(top, structure=np.ones((3, 3)))
    peak = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
    rows, cols = np.nonzero(labels == labels[peak])
    (map_h, map_w), (width, height) = anomaly_map.shape, size
    sides, shifts = [], []
    for lo, hi, cells, full in ((cols.min(), cols.max() + 1, map_w, width), (rows.min(), rows.max() + 1, map_h, height)):
        start, end = lo * full / cells, hi * full / cells
        side = max((end - start) * box_scale, full / grid)
        centre = (start + end) / 2
        first, last = floor(centre - side / 2), ceil(centre + side / 2)
        shift = 0
        if last - first >= full:
            first, last = 0, full
        elif first < 0:
            shift = -first
        elif last > full:
            shift = full - last
        sides.append((first + shift, last + shift))
        shifts.append(shift)
    (left, right), (upper, lower) = sides
    return (left, upper, right, lower), (shifts[0], shifts[1])


def downscale(image: Image.Image, factor: int) -> Image.Image:
    return image.resize((image.width // factor, image.height // factor), Image.Resampling.BICUBIC)


def _self_check() -> None:
    assert JET[0].tolist() == [0, 0, 128] and JET[255].tolist() == [128, 0, 0]

    def blob(rows: slice, cols: slice) -> np.ndarray:
        # each blob covers just over 1% of the pixels, so the top pixels are the blob itself
        m = np.zeros((518, 518), dtype=np.float32)
        m[rows, cols] = 1
        return m

    def box_of(m: np.ndarray) -> tuple:
        return zoom_box(m, (900, 600), 0.01, 2, 3)

    # near the centre: widened to one grid cell, not moved
    assert box_of(blob(slice(233, 285), slice(233, 285))) == ((300, 200, 600, 400), (0, 0))
    # top-left corner: moved right and down with its size kept
    assert box_of(blob(slice(0, 52), slice(0, 52))) == ((0, 0, 301, 201), (105, 70))
    # bottom-right corner: moved left and up with its size kept
    assert box_of(blob(slice(466, 518), slice(466, 518))) == ((599, 399, 900, 600), (-105, -70))
    # a band wider than half the image: the doubled width is cut to the whole image
    assert box_of(blob(slice(250, 268), slice(0, 518))) == ((0, 200, 900, 400), (0, 0))

    image = jet_image(blob(slice(0, 52), slice(0, 52)), (900, 600), 0.0, 1.0)
    assert image.size == (900, 600) and image.getpixel((0, 0)) == (128, 0, 0)
    assert image.getpixel((899, 599)) == (0, 0, 128)
    assert downscale(image, 2).size == (450, 300)
    print("maps self-check ok")


if __name__ == "__main__":
    _self_check()
