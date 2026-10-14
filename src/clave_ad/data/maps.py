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


def _fit(centre: float, side: float, full: int) -> tuple[int, int, int]:
    """Span (first, last) of the given side about centre on an axis of length full, and how far it was moved.

    A span crossing the edge is moved inside with its length kept; a span longer than the axis is cut to it (shift 0).
    """
    first, last = floor(centre - side / 2), ceil(centre + side / 2)
    if last - first >= full:
        return 0, full, 0
    shift = -first if first < 0 else (full - last if last > full else 0)
    return first + shift, last + shift, shift


def peak(anomaly_map: np.ndarray) -> tuple[int, int]:
    """(row, col) of the first maximum in row-major order."""
    row, col = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
    return int(row), int(col)


def zoom_box(anomaly_map: np.ndarray, size: tuple[int, int], top_fraction: float, box_scale: float,
             grid: int) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Zoom-in crop (left, top, right, bottom) in original pixels, and how far it was moved (dx, dy) to fit the image.

    At map resolution: the 8-connected component of the top pixels that holds the peak. Its bounding box is moved to
    original coordinates, scaled about its centre and widened to at least one grid cell. A box crossing the image edge
    is moved inside with its size kept; an axis longer than the image is cut to the whole image (shift 0).
    """
    top = anomaly_map >= np.percentile(anomaly_map, 100 * (1 - top_fraction))
    labels, _ = ndimage.label(top, structure=np.ones((3, 3)))
    rows, cols = np.nonzero(labels == labels[peak(anomaly_map)])
    (map_h, map_w), (width, height) = anomaly_map.shape, size
    spans = []
    for lo, hi, cells, full in ((cols.min(), cols.max() + 1, map_w, width), (rows.min(), rows.max() + 1, map_h, height)):
        start, end = lo * full / cells, hi * full / cells
        spans.append(_fit((start + end) / 2, max((end - start) * box_scale, full / grid), full))
    (left, right, dx), (upper, lower, dy) = spans
    return (left, upper, right, lower), (dx, dy)


def type_box(anomaly_map: np.ndarray, size: tuple[int, int], crop_cells: float,
             grid: int) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Type verification crop (A4): crop_cells grid cells per side, centred on the peak pixel's centre in original
    coordinates, moved inside the image at the edge with its size kept. Returns the box and the shift (dx, dy)."""
    row, col = peak(anomaly_map)
    (map_h, map_w), (width, height) = anomaly_map.shape, size
    left, right, dx = _fit((col + 0.5) * width / map_w, crop_cells * width / grid, width)
    upper, lower, dy = _fit((row + 0.5) * height / map_h, crop_cells * height / grid, height)
    return (left, upper, right, lower), (dx, dy)


def cell_shares(anomaly_map: np.ndarray, top_fraction: float, grid: int) -> dict[str, float]:
    """Share of the top map pixels in each grid cell ("r1c1" ... ), a pixel counted in the cell holding its centre."""
    top = anomaly_map >= np.percentile(anomaly_map, 100 * (1 - top_fraction))
    rows, cols = np.nonzero(top)
    map_h, map_w = anomaly_map.shape
    cell_rows = np.floor((rows + 0.5) * grid / map_h).astype(int)
    cell_cols = np.floor((cols + 0.5) * grid / map_w).astype(int)
    counts = np.zeros((grid, grid))
    np.add.at(counts, (cell_rows, cell_cols), 1)
    counts /= counts.sum()
    return {f"r{i + 1}c{j + 1}": float(counts[i, j]) for i in range(grid) for j in range(grid)}


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

    # type crop: 1.5 cells about the peak pixel, moved inside at the edge
    single = np.zeros((518, 518), dtype=np.float32)
    single[259, 259] = 1
    assert type_box(single, (900, 600), 1.5, 3) == ((225, 150, 676, 451), (0, 0))
    single[:] = 0
    single[0, 0] = 1
    assert type_box(single, (900, 600), 1.5, 3) == ((0, 0, 451, 301), (225, 150))
    shares = cell_shares(blob(slice(233, 285), slice(233, 285)), 0.01, 3)
    assert shares["r2c2"] == 1.0 and sum(shares.values()) == 1.0

    image = jet_image(blob(slice(0, 52), slice(0, 52)), (900, 600), 0.0, 1.0)
    assert image.size == (900, 600) and image.getpixel((0, 0)) == (128, 0, 0)
    assert image.getpixel((899, 599)) == (0, 0, 128)
    assert downscale(image, 2).size == (450, 300)
    print("maps self-check ok")


if __name__ == "__main__":
    _self_check()
