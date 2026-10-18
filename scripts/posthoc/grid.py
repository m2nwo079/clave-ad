"""3x3 grid lines drawn on the original image for the post-hoc E1 condition (configs/posthoc/c4_cause.yaml).

Line positions follow the cell boundaries floor(k * side / size), the same split as eval/claims.py cell_boxes;
report.py checks that locally. No cell labels and no anti-aliasing.
"""

from math import floor

from PIL import Image, ImageDraw


def boundaries(side: int, size: int) -> list[int]:
    """Inner cell boundaries along one side."""
    return [floor(k * side / size) for k in range(1, size)]


def line_width(image_size: tuple[int, int], width_share: float, min_px: int) -> int:
    return max(min_px, round(min(image_size) * width_share))


def draw_grid(image: Image.Image, size: int, width_share: float, min_px: int, border_px: int) -> Image.Image:
    """Copy of the image with white grid lines, a black border of border_px on each side, on the cell boundaries."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    w = line_width(out.size, width_share, min_px)
    for x in boundaries(width, size):
        left = x - w // 2
        draw.rectangle([left - border_px, 0, left + w - 1 + border_px, height - 1], fill="black")
        draw.rectangle([left, 0, left + w - 1, height - 1], fill="white")
    for y in boundaries(height, size):
        top = y - w // 2
        draw.rectangle([0, top - border_px, width - 1, top + w - 1 + border_px], fill="black")
        draw.rectangle([0, top, width - 1, top + w - 1], fill="white")
    # vertical lines cross horizontal ones, so the white of the earlier lines is restored where black was drawn over
    for x in boundaries(width, size):
        left = x - w // 2
        draw.rectangle([left, 0, left + w - 1, height - 1], fill="white")
    return out


def _self_check() -> None:
    assert boundaries(1284, 3) == [428, 856] and boundaries(10, 3) == [3, 6]
    assert line_width((1284, 1168), 0.003, 2) == 4 and line_width((300, 200), 0.003, 2) == 2
    grey = Image.new("RGB", (300, 200), (128, 128, 128))
    out = draw_grid(grey, 3, 0.003, 2, 1)
    px = out.load()
    # x boundary 100, width 2: white at 99..100, black at 98 and 101, untouched further out
    assert px[99, 50] == px[100, 50] == (255, 255, 255)
    assert px[98, 50] == px[101, 50] == (0, 0, 0) and px[97, 50] == (128, 128, 128)
    # y boundary 66, and crossings stay white
    assert px[50, 65] == px[50, 66] == (255, 255, 255) and px[50, 64] == (0, 0, 0)
    assert px[100, 66] == (255, 255, 255)
    # only white, black and the original grey: no anti-aliasing
    assert {c for _, c in out.getcolors(maxcolors=10)} == {(255, 255, 255), (0, 0, 0), (128, 128, 128)}
    assert grey.getpixel((100, 50)) == (128, 128, 128)
    # a wider border reaches one pixel further on each side
    wide = draw_grid(grey, 3, 0.003, 2, 2).load()
    assert wide[97, 50] == wide[102, 50] == (0, 0, 0) and wide[96, 50] == (128, 128, 128)
    print("grid self-check ok")


if __name__ == "__main__":
    _self_check()
