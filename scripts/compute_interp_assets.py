"""Compute static interpretability assets for the browser demo.

Outputs:
  demo/op_units.json       — per-operator unit tags ranked by purity
  demo/op_directions.json  — pairwise SVD steering directions + INLP basis

Uses demo/encode.onnx (no checkpoint required). Inputs are rendered
glyphs that match how the JS demo draws operators (Arial bold) and how
the training pipeline normalized MNIST (mean 0.1307, std 0.3081).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as rt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MEAN, STD = 0.1307, 0.3081
OPS = ["+", "−", "×", "÷"]
DIGITS = [str(d) for d in range(10)]

# Render a centred 28×28 glyph using PIL's default bitmap font scaled via
# a 280×280 canvas then downsampled. This mirrors the JS pad pipeline:
# draw at 280×280, find bbox, recentre into the 20×20 inner region of a
# 28×28 image, MNIST-normalize.
def _pick_font(size: int) -> ImageFont.ImageFont:
    for name in ("Arial Bold.ttf", "Arial.ttf", "Helvetica.ttc", "Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_glyph(text: str) -> np.ndarray:
    big = Image.new("L", (280, 280), 0)
    d = ImageDraw.Draw(big)
    font = _pick_font(200)
    bbox = d.textbbox((0, 0), text, font=font, anchor="mm")
    d.text((140, 140 + 4), text, fill=255, font=font, anchor="mm")
    arr = np.asarray(big, dtype=np.float32)
    ys, xs = np.where(arr > 16)
    if len(xs) == 0:
        out = np.full((28, 28), -MEAN / STD, dtype=np.float32)
        return out
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    side = max(max_x - min_x + 1, max_y - min_y + 1)
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    half = side / 2 + 8
    crop = big.crop((cx - half, cy - half, cx + half, cy + half))
    inner = crop.resize((20, 20), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(inner, (4, 4))
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    return (a - MEAN) / STD


def blank() -> np.ndarray:
    return np.full((28, 28), -MEAN / STD, dtype=np.float32)


def encode_batch(sess: rt.InferenceSession, triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Returns [N, 256] trunk vectors."""
    out = []
    for a, o, b in triples:
        feed = {
            "img_a": a[None, None].astype(np.float32),
            "img_op": o[None, None].astype(np.float32),
            "img_b": b[None, None].astype(np.float32),
        }
        t = sess.run(["mul_out_1"], feed)[0][0]
        out.append(t)
    return np.stack(out)


def main() -> None:
    enc_path = ROOT / "demo" / "encode.onnx"
    sess = rt.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])

    print("rendering glyphs…")
    digit_imgs = {d: render_glyph(d) for d in DIGITS}
    op_imgs = {op: render_glyph(op) for op in OPS}

    # Build the 4×10×10 grid: for each operator, encode every (digit_a, digit_b) pair.
    print("encoding 4 × 10 × 10 grid…")
    grid = {}  # op -> [10, 10, 256]
    for op in OPS:
        rows = []
        for a in DIGITS:
            row = []
            for b in DIGITS:
                t = encode_batch(sess, [(digit_imgs[a], op_imgs[op], digit_imgs[b])])[0]
                row.append(t)
            rows.append(np.stack(row))
        grid[op] = np.stack(rows)
        print(f"  {op}: {grid[op].shape}, mean act = {grid[op].mean():.3f}")

    # Operator unit tags: for each unit u, average activation per operator,
    # tag with argmax. Purity = (winner - runner-up) / winner.
    mean_per_op = np.stack([grid[op].reshape(-1, 256).mean(0) for op in OPS])  # [4, 256]
    winner = mean_per_op.argmax(0)  # [256]
    sorted_acts = np.sort(mean_per_op, axis=0)  # ascending
    purity = (sorted_acts[-1] - sorted_acts[-2]) / (sorted_acts[-1] + 1e-8)
    purity = np.where(sorted_acts[-1] > 1e-6, purity, 0.0)

    op_units = {op: [] for op in OPS}
    for u in range(256):
        op = OPS[int(winner[u])]
        op_units[op].append({"u": int(u), "p": float(purity[u]), "act": float(mean_per_op[int(winner[u]), u])})
    for op in OPS:
        op_units[op].sort(key=lambda r: -r["p"])
    counts = {op: len(op_units[op]) for op in OPS}
    print(f"per-op unit counts: {counts}")

    # SVD pairwise directions: t(j) - t(i) flattened over 10×10
    print("computing pairwise SVD directions…")
    directions = {}
    for i, oi in enumerate(OPS):
        for j, oj in enumerate(OPS):
            if i == j:
                continue
            delta = (grid[oj] - grid[oi]).reshape(-1, 256)  # [100, 256]
            U, S, Vt = np.linalg.svd(delta, full_matrices=False)
            v = Vt[0]  # top-1 right singular vector
            # Sign so that adding v moves t toward oj
            sign = np.sign(delta.mean(0) @ v)
            v = (v * (sign if sign != 0 else 1.0)).astype(np.float32)
            directions[f"{oi}->{oj}"] = {
                "v": v.tolist(),
                "sv": float(S[0]),
                "ratio": float(S[0] / (S.sum() + 1e-8)),
            }

    # INLP basis: stack operator centroids, subtract grand centroid, SVD.
    print("computing INLP operator subspace…")
    centroids = np.stack([grid[op].reshape(-1, 256).mean(0) for op in OPS])  # [4, 256]
    centered = centroids - centroids.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    inlp_basis = Vt.astype(np.float32)  # [<=4, 256]
    inlp_sv = S.astype(np.float32)

    # Mean trunk (handy baseline for "reset")
    mean_t = np.stack([grid[op].reshape(-1, 256).mean(0) for op in OPS]).mean(0)

    # Save
    (ROOT / "demo" / "op_units.json").write_text(json.dumps({
        "units": op_units,
        "ops": OPS,
        "counts": counts,
        "mean_per_op": mean_per_op.tolist(),  # for the cumulative slider ranking
    }, indent=0))
    (ROOT / "demo" / "op_directions.json").write_text(json.dumps({
        "ops": OPS,
        "pairs": directions,
        "inlp_basis": inlp_basis.tolist(),
        "inlp_sv": inlp_sv.tolist(),
        "mean_t": mean_t.tolist(),
        "op_centroids": centroids.tolist(),
    }, indent=0))
    print("wrote demo/op_units.json and demo/op_directions.json")

    # Quick sanity: steering verification
    print()
    print("steering sanity:  add λ·v_(+→×)  to t(+, 3, 4) and report shift")
    base = grid["+"][3, 4]
    v = np.asarray(directions["+->×"]["v"], dtype=np.float32)
    for lam in [0.0, 0.5, 1.0, 2.0]:
        new = base + lam * v
        cos_to_plus = float(centroids[0] @ new / (np.linalg.norm(centroids[0]) * np.linalg.norm(new) + 1e-8))
        cos_to_times = float(centroids[2] @ new / (np.linalg.norm(centroids[2]) * np.linalg.norm(new) + 1e-8))
        print(f"  λ={lam:.1f}  cos(t, +centroid)={cos_to_plus:.3f}  cos(t, ×centroid)={cos_to_times:.3f}")


if __name__ == "__main__":
    main()
