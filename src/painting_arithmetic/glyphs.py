"""Symbol glyph rendering with robust font selection.

We render four operator glyphs (``+ − × ÷``) and ten digit glyphs as 28×28
white-on-black images, plus 28×84 result images split into three slots
``[sign | tens | units]``.

Font selection is **glyph-coverage-aware**: we pick the first font from a
candidate list that actually has every character we need, falling back to
matplotlib's bundled DejaVu Sans which ships with the matplotlib package and
is therefore available on every platform. This avoids the silent
".notdef" / tofu-rectangle issue when a system font lacks ``−`` or ``÷``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import matplotlib as _mpl
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OP_GLYPHS = ["+", "−", "×", "÷"]
OPS = ["+", "-", "*", "//"]               # internal names
OP_LABEL_OFFSET = 10
NUM_SYMBOLS = 14                          # 10 digits + 4 operators

RESULT_MIN, RESULT_MAX = -9, 81           # full range for single-digit ops
NUM_RESULTS = RESULT_MAX - RESULT_MIN + 1 # 91 distinct integers

IMG_H, IMG_W = 28, 84
SLOT_W = 28

# Matplotlib ships its own DejaVu Sans, guaranteed to be on disk.
_MPL_DEJAVU = str(Path(_mpl.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")

FONT_CANDIDATES = [
    _MPL_DEJAVU,
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

REQUIRED_CHARS = OP_GLYPHS + [str(d) for d in range(10)]


# ---------------------------------------------------------------------------
# Font selection
# ---------------------------------------------------------------------------


def _font_has_glyph(font: ImageFont.FreeTypeFont, char: str) -> bool:
    """Compare ``char`` rendering against U+E000 (Private-Use, no font assigns
    a glyph there). Identical pixels mean the font is showing ``.notdef``."""
    a = Image.new("L", (80, 80), 0)
    b = Image.new("L", (80, 80), 0)
    ImageDraw.Draw(a).text((5, 5), char, fill=255, font=font)
    ImageDraw.Draw(b).text((5, 5), "", fill=255, font=font)
    return np.asarray(a).tobytes() != np.asarray(b).tobytes()


def _find_good_font_path(probe_size: int = 40) -> str:
    """Walk the candidate list and return the first path whose font covers all
    of REQUIRED_CHARS. Fall back to the first usable path with a warning."""
    for path in FONT_CANDIDATES:
        if not path:
            continue
        try:
            f = ImageFont.truetype(path, probe_size)
        except (OSError, IOError):
            continue
        if all(_font_has_glyph(f, c) for c in REQUIRED_CHARS):
            return path
    for path in FONT_CANDIDATES:
        if not path:
            continue
        try:
            ImageFont.truetype(path, probe_size)
            return path
        except (OSError, IOError):
            continue
    raise RuntimeError("No TrueType font found on this system.")


_SELECTED_FONT_PATH: Optional[str] = None


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    global _SELECTED_FONT_PATH
    if _SELECTED_FONT_PATH is None:
        _SELECTED_FONT_PATH = _find_good_font_path()
    return ImageFont.truetype(_SELECTED_FONT_PATH, size)


def selected_font_path() -> str:
    """Return the path of the font selected on first use (for logging)."""
    if _SELECTED_FONT_PATH is None:
        _load_font(40)
    return _SELECTED_FONT_PATH  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Glyph rendering — MNIST-style preprocessing
# ---------------------------------------------------------------------------


def _render_centered(
    text: str,
    *,
    size: int = 28,
    target_max: int = 22,
    src_font: int = 120,
    src_canvas: int = 200,
) -> Image.Image:
    """Render ``text`` at high resolution, crop to the actual non-zero pixel
    bbox, scale so the longest side is ``target_max``, centre in a black
    ``size``×``size`` frame.

    This mirrors how MNIST itself was preprocessed: the rendered character
    fills a known fraction of the frame, independent of which font we ended
    up using.
    """
    img = Image.new("L", (src_canvas, src_canvas), 0)
    draw = ImageDraw.Draw(img)
    font = _load_font(src_font)
    bb = draw.textbbox((0, 0), text, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    x = (src_canvas - w) // 2 - bb[0]
    y = (src_canvas - h) // 2 - bb[1]
    draw.text((x, y), text, fill=255, font=font)

    arr = np.asarray(img)
    nz = arr > 16
    if not nz.any():
        return Image.new("L", (size, size), 0)
    rows = np.where(nz.any(axis=1))[0]
    cols = np.where(nz.any(axis=0))[0]
    cropped = img.crop((int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1))
    cw, ch = cropped.size
    scale = target_max / max(cw, ch)
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    cropped = cropped.resize((nw, nh), Image.LANCZOS)
    out = Image.new("L", (size, size), 0)
    out.paste(cropped, ((size - nw) // 2, (size - nh) // 2))
    return out


def render_glyph_clean(op_idx: int) -> Image.Image:
    """Canonical (un-augmented) operator glyph."""
    return _render_centered(OP_GLYPHS[op_idx], size=28, target_max=22)


def render_glyph_augmented(op_idx: int, rng: random.Random) -> Image.Image:
    """Augmented operator glyph — random target size, rotation, jitter, blur."""
    base = _render_centered(OP_GLYPHS[op_idx], size=28, target_max=rng.randint(18, 24))
    rotation = rng.uniform(-15.0, 15.0)
    if rotation != 0:
        base = base.rotate(rotation, resample=Image.BILINEAR)
    dx = rng.randint(-3, 3)
    dy = rng.randint(-3, 3)
    if dx or dy:
        shifted = Image.new("L", (28, 28), 0)
        shifted.paste(base, (dx, dy))
        base = shifted
    blur = rng.choice([0.0, 0.0, 0.5])
    if blur > 0:
        base = base.filter(ImageFilter.GaussianBlur(blur))
    return base


def render_digit_glyph(d: int) -> Image.Image:
    """A 28×28 glyph for one digit, MNIST-style preprocessing."""
    return _render_centered(str(d), size=28, target_max=22)


def render_minus_glyph() -> Image.Image:
    """A 28×28 minus glyph for the sign slot."""
    return _render_centered("−", size=28, target_max=18)


# ---------------------------------------------------------------------------
# Result image — 28×84 with three slots
# ---------------------------------------------------------------------------


def render_result_image(result: int) -> np.ndarray:
    """28×84 uint8 image of ``result`` rendered as ``[sign | tens | units]``."""
    if not RESULT_MIN <= result <= RESULT_MAX:
        raise ValueError(f"result out of bounds: {result}")
    arr = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    if result < 0:
        arr[:, 0:SLOT_W] = np.asarray(render_minus_glyph(), dtype=np.uint8)
        absr = -result
    else:
        absr = result
    if absr >= 10:
        arr[:, SLOT_W : 2 * SLOT_W] = np.asarray(render_digit_glyph(absr // 10), dtype=np.uint8)
    arr[:, 2 * SLOT_W : 3 * SLOT_W] = np.asarray(render_digit_glyph(absr % 10), dtype=np.uint8)
    return arr


def render_all_results() -> np.ndarray:
    """(91, 28, 84) float32 stack of canonical target images in [0, 1]."""
    return np.stack(
        [render_result_image(r).astype(np.float32) / 255.0 for r in range(RESULT_MIN, RESULT_MAX + 1)]
    )


def slot_labels(result: int) -> tuple[int, int, int]:
    """Decompose ``result`` into ``(sign_label, tens_label, units_label)`` for
    auxiliary classification supervision."""
    sign = 1 if result < 0 else 0
    absr = abs(result)
    return sign, absr // 10, absr % 10


__all__ = [
    "OP_GLYPHS",
    "OPS",
    "OP_LABEL_OFFSET",
    "NUM_SYMBOLS",
    "RESULT_MIN",
    "RESULT_MAX",
    "NUM_RESULTS",
    "IMG_H",
    "IMG_W",
    "SLOT_W",
    "FONT_CANDIDATES",
    "selected_font_path",
    "render_glyph_clean",
    "render_glyph_augmented",
    "render_digit_glyph",
    "render_minus_glyph",
    "render_result_image",
    "render_all_results",
    "slot_labels",
]
