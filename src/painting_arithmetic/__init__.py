"""Painting Arithmetic — a tiny rational-kernel network for visual symbolic computation."""

from . import data, glyphs, losses, model, viz
from .model import YatArithmeticGen

__version__ = "0.1.0"
__all__ = ["data", "glyphs", "losses", "model", "viz", "YatArithmeticGen"]
