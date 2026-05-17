"""Sanity tests for glyph rendering and font selection."""

import numpy as np
import pytest

from painting_arithmetic import glyphs


def test_font_covers_required_chars():
    """The selected font must have a real glyph for every operator and digit."""
    from PIL import ImageFont
    path = glyphs.selected_font_path()
    font = ImageFont.truetype(path, 40)
    for c in glyphs.REQUIRED_CHARS:
        assert glyphs._font_has_glyph(font, c), f"font {path} missing glyph for {c!r}"


def test_render_glyph_clean_returns_28x28():
    for i in range(4):
        img = np.asarray(glyphs.render_glyph_clean(i))
        assert img.shape == (28, 28)
        assert img.dtype == np.uint8
        # The image must be non-trivial — at least a few white pixels.
        assert (img > 0).sum() > 10


def test_render_result_image_shape_and_range():
    arr = glyphs.render_result_image(42)
    assert arr.shape == (28, 84)
    assert arr.min() == 0 and arr.max() == 255

    arr_neg = glyphs.render_result_image(-9)
    # Sign slot non-empty.
    assert (arr_neg[:, :28] > 0).any()


def test_render_all_results_stack():
    stack = glyphs.render_all_results()
    assert stack.shape == (91, 28, 84)
    assert stack.dtype == np.float32
    assert stack.min() == 0.0
    assert stack.max() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("r,expected", [
    (0,  (0, 0, 0)),
    (5,  (0, 0, 5)),
    (10, (0, 1, 0)),
    (42, (0, 4, 2)),
    (81, (0, 8, 1)),
    (-9, (1, 0, 9)),
    (-1, (1, 0, 1)),
])
def test_slot_labels(r, expected):
    assert glyphs.slot_labels(r) == expected
