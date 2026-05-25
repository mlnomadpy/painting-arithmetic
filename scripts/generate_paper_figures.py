"""Generate the four paper figures that aren't produced by experiments/.

Outputs (saved into viz_assets/):
  trunk_pca_v3.png          — PCA of trunk vectors, coloured by result and by op
  op_as_shift.png           — SVD spectrum of pairwise operator differences
  decoder_unit_atlas.png    — top-24 trunk units painted by decoder
  prototype_gallery_v3.png  — per-unit prototype winners with model output

Uses only the deployed ONNX checkpoints (demo/encode.onnx + demo/decode.onnx),
real MNIST samples for digit operands, and Arial-rendered operator glyphs
that match the training pipeline.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as rt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "viz_assets"
OUT.mkdir(exist_ok=True)

MEAN, STD = 0.1307, 0.3081
OPS = ["+", "-", "x", "/"]
OP_GLYPHS = ["+", "−", "×", "÷"]
OP_COLORS = ["#D87F26", "#2E7D8A", "#8B3A62", "#5B7C3E"]
H = 256
SLOT_DIM = 64
DIGITS_PER_CLASS = 8     # number of MNIST samples averaged per digit for the grid
LIB_DIGIT_SAMPLES = 8    # library size per digit class
LIB_OP_SAMPLES = 4       # library size per operator class


# ---------------------------------------------------------------------------
# MNIST + glyph rendering
# ---------------------------------------------------------------------------


def load_mnist_test() -> tuple[np.ndarray, np.ndarray]:
    """Returns (10000, 28, 28) uint8 images and (10000,) labels."""
    raw = ROOT / "data" / "MNIST" / "raw"
    with gzip.open(raw / "t10k-images-idx3-ubyte.gz", "rb") as f:
        f.read(16)
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    with gzip.open(raw / "t10k-labels-idx1-ubyte.gz", "rb") as f:
        f.read(8)
        labs = np.frombuffer(f.read(), dtype=np.uint8)
    return imgs, labs


def bin_by_digit(imgs: np.ndarray, labs: np.ndarray) -> list[np.ndarray]:
    """Returns a length-10 list of (n_d, 28, 28) float32 arrays normalised."""
    bins = []
    for d in range(10):
        sel = imgs[labs == d].astype(np.float32) / 255.0
        bins.append((sel - MEAN) / STD)
    return bins


def _pick_font(size: int) -> ImageFont.ImageFont:
    for name in ("Arial Bold.ttf", "Arial.ttf", "Helvetica.ttc", "Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_glyph(text: str, clean: bool = True, rng: np.random.Generator | None = None) -> np.ndarray:
    big = Image.new("L", (280, 280), 0)
    d = ImageDraw.Draw(big)
    font = _pick_font(200)
    d.text((140, 144), text, fill=255, font=font, anchor="mm")
    arr = np.asarray(big, dtype=np.float32)
    ys, xs = np.where(arr > 16)
    if len(xs) == 0:
        return np.full((28, 28), -MEAN / STD, dtype=np.float32)
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    side = max(max_x - min_x + 1, max_y - min_y + 1)
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    half = side / 2 + 8
    crop = big.crop((cx - half, cy - half, cx + half, cy + half)).resize((20, 20), Image.LANCZOS)
    out = Image.new("L", (28, 28), 0)
    out.paste(crop, (4, 4))
    arr28 = np.asarray(out, dtype=np.float32) / 255.0
    return (arr28 - MEAN) / STD


# ---------------------------------------------------------------------------
# ONNX wrappers
# ---------------------------------------------------------------------------


def load_sessions():
    enc = rt.InferenceSession(str(ROOT / "demo" / "encode.onnx"), providers=["CPUExecutionProvider"])
    dec = rt.InferenceSession(str(ROOT / "demo" / "decode.onnx"), providers=["CPUExecutionProvider"])
    return enc, dec


def load_h1_weight() -> np.ndarray:
    """Returns the (256, 192) h_1 weight matrix from encode.onnx."""
    model = onnx.load(str(ROOT / "demo" / "encode.onnx"))
    for init in model.graph.initializer:
        if list(init.dims) == [192, 256]:
            w = np.frombuffer(init.raw_data, dtype=np.float32).reshape(192, 256)
            return w.T.copy()  # → (256, 192)
    raise RuntimeError("could not find 192×256 weight in encode.onnx")


def encode_batch(enc: rt.InferenceSession, a: np.ndarray, op: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Each input is (n, 28, 28) float32; returns (n, 256) trunk vectors.

    Loops one-at-a-time because the deployed ONNX has a fixed batch=1.
    """
    n = a.shape[0]
    out = np.empty((n, H), dtype=np.float32)
    for i in range(n):
        ta = a[i:i+1, None, :, :]
        to = op[i:i+1, None, :, :]
        tb = b[i:i+1, None, :, :]
        out[i] = enc.run(None, {"img_a": ta, "img_op": to, "img_b": tb})[0][0]
    return out


def encode_single_slot(enc: rt.InferenceSession, x: np.ndarray) -> np.ndarray:
    """Encode an (n, 28, 28) batch through the *symbol encoder only* by feeding
    the same image to all three slots and reading one of the 64-d sub-embeddings.

    The shared encoder runs three times per forward pass; the three outputs are
    identical when the three inputs are identical. We approximate the symbol
    embedding for ``x`` as the first 64-d slot of the fused trunk-input."""
    # We use the fact that fused = concat(e_a, e_op, e_b); since the encoder is
    # shared, e_a = e_op = e_b when img_a = img_op = img_b. The fused vector
    # then has e at positions [0:64]. We reconstruct this by running the trunk
    # and reading the input. But encode.onnx already maps fused → t, so we
    # need to recover the fused intermediate.
    #
    # Cleanest workaround: load the h_1 weight separately (see load_h1_weight)
    # and inspect the *input* to that op via a partial forward. Easier still:
    # we approximate by feeding a blank to img_a / img_b and putting `x` only
    # in img_op, then exploit the symmetry — but that changes the trunk.
    #
    # The simplest viable approach: encode each x with all three slots = x,
    # then estimate the per-slot embedding by averaging the three blocks of
    # the *Linear* before the Yat. Since we don't have that intermediate
    # exposed, we use a separate matmul: build a per-image "embedding" by
    # running the cheap onnxruntime extract of the GAP+Linear chain.
    #
    # In practice we don't need exact embeddings for the prototype gallery;
    # the qualitative winners come out fine using the trunk vector itself as
    # the comparison space. So return the trunk run with all three slots = x.
    n = x.shape[0]
    out = np.empty((n, H), dtype=np.float32)
    for i in range(n):
        t = x[i:i+1, None, :, :]
        out[i] = enc.run(None, {"img_a": t, "img_op": t, "img_b": t})[0][0]
    return out


def decode_batch(dec: rt.InferenceSession, t: np.ndarray) -> np.ndarray:
    """t: (n, 256). Returns (n, 28, 84) painted images."""
    n = t.shape[0]
    out = np.empty((n, 28, 84), dtype=np.float32)
    for i in range(n):
        out[i] = dec.run(None, {"t": t[i:i+1]})[0][0, 0]
    return out


# ---------------------------------------------------------------------------
# Grid collection
# ---------------------------------------------------------------------------


def collect_grid(enc: rt.InferenceSession, bins: list[np.ndarray]) -> dict:
    """For each (a, op, b) cell on the canonical 4×10×10 grid, average the
    trunk vector across ``DIGITS_PER_CLASS`` MNIST samples per operand digit
    to reduce per-sample noise.

    Returns: trunks (N, 256), results (N,), ops (N,), a_idx, b_idx, with
    division-by-zero removed. N = 4·10·10 − 10 = 390.
    """
    n = DIGITS_PER_CLASS
    op_imgs = np.stack([render_glyph(g) for g in OP_GLYPHS], axis=0)  # (4, 28, 28)
    trunks, results, ops_out, a_idx, b_idx = [], [], [], [], []
    for op in range(4):
        for a in range(10):
            for b in range(10):
                if op == 3 and b == 0:
                    continue
                # Pair n samples of digit a with n samples of digit b.
                a_imgs = bins[a][:n]
                b_imgs = bins[b][:n]
                o_img = op_imgs[op]
                # Cross-product n×n is overkill; use n diagonal pairs.
                a_arr = a_imgs
                b_arr = b_imgs
                o_arr = np.broadcast_to(o_img, (n, 28, 28)).copy()
                ts = encode_batch(enc, a_arr, o_arr, b_arr)  # (n, 256)
                trunks.append(ts.mean(axis=0))
                ops_out.append(op); a_idx.append(a); b_idx.append(b)
                if op == 0: results.append(a + b)
                elif op == 1: results.append(a - b)
                elif op == 2: results.append(a * b)
                else: results.append(a // b)
    return {
        "trunks": np.stack(trunks, axis=0),
        "results": np.array(results),
        "ops": np.array(ops_out),
        "a_idx": np.array(a_idx),
        "b_idx": np.array(b_idx),
    }


# ---------------------------------------------------------------------------
# 1. trunk PCA
# ---------------------------------------------------------------------------


def fig_trunk_pca(grid: dict, out_path: Path):
    X = grid["trunks"]
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    vr = (S ** 2 / (S ** 2).sum())[:2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=grid["results"],
                         cmap="turbo", s=24, alpha=0.85, edgecolor="none")
    axes[0].set_title(f"trunk PCA — coloured by result\nPC1 = {vr[0]*100:.1f}%   PC2 = {vr[1]*100:.1f}%")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    plt.colorbar(sc, ax=axes[0], label="result", shrink=0.85)
    for i in range(4):
        m = grid["ops"] == i
        axes[1].scatter(coords[m, 0], coords[m, 1], c=OP_COLORS[i], s=24, alpha=0.75,
                        edgecolor="none", label=OP_GLYPHS[i])
    axes[1].set_title("same scatter, coloured by operator")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].legend(loc="best", frameon=True)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {out_path.name}")


# ---------------------------------------------------------------------------
# 2. operator-as-shift SVD
# ---------------------------------------------------------------------------


def fig_op_as_shift(grid: dict, out_path: Path):
    by_op = {i: grid["trunks"][grid["ops"] == i] for i in range(4)}
    pairs = [(0, 1, "+ vs −"), (0, 2, "+ vs ×"), (0, 3, "+ vs ÷"),
             (1, 2, "− vs ×"), (1, 3, "− vs ÷"), (2, 3, "× vs ÷")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (i, j, label) in zip(axes.ravel(), pairs):
        n = min(by_op[i].shape[0], by_op[j].shape[0])
        diff = by_op[i][:n] - by_op[j][:n]
        _, S, _ = np.linalg.svd(diff, full_matrices=False)
        vr = (S ** 2) / (S ** 2).sum()
        ax.bar(range(1, 21), vr[:20], color="#2E7D8A", edgecolor="none")
        ax.set_title(f"{label}\ntop-1 = {vr[0]*100:.1f}%   top-3 = {vr[:3].sum()*100:.1f}%",
                     fontsize=10)
        ax.set_xlabel("singular rank"); ax.set_ylabel("variance ratio")
        ax.set_ylim(0, max(0.85, vr[0] + 0.05))
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("operator-as-shift SVD spectrum (10×10 (a,b) grid)", y=1.01, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {out_path.name}")


# ---------------------------------------------------------------------------
# 3. decoder unit atlas
# ---------------------------------------------------------------------------


def fig_decoder_unit_atlas(dec: rt.InferenceSession, out_path: Path, alpha: float = 3.0, n_show: int = 24):
    baseline = dec.run(None, {"t": np.zeros((1, H), dtype=np.float32)})[0][0, 0]
    foot = np.zeros((H, 28, 84), dtype=np.float32)
    for u in range(H):
        v = np.zeros((1, H), dtype=np.float32); v[0, u] = alpha
        img = dec.run(None, {"t": v})[0][0, 0]
        foot[u] = img - baseline
    norms = np.sqrt((foot ** 2).sum(axis=(1, 2)))
    top = np.argsort(-norms)[:n_show]
    rows, cols = 4, 6
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.8))
    vmax = float(np.max(np.abs(foot[top])))
    for i, u in enumerate(top):
        ax = axes[i // cols, i % cols]
        ax.imshow(foot[u], cmap="seismic", vmin=-vmax, vmax=vmax, aspect="equal",
                  interpolation="bilinear")
        ax.set_title(f"unit {int(u)}   ‖Δ‖ = {norms[u]:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"decoder unit atlas — top {n_show} trunk units by L2 footprint\n"
                 f"red = adds bright pixels · blue = subtracts (α = {alpha})",
                 fontsize=11, y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {out_path.name}")


# ---------------------------------------------------------------------------
# 4. prototype gallery — per-slot blank-out approach
# ---------------------------------------------------------------------------


def _build_library(bins):
    """Returns (lib_imgs [N, 28, 28], lib_labels [N]) where labels are
    0–9 for digits and 10–13 for + − × ÷."""
    pieces_imgs, pieces_labels = [], []
    for d in range(10):
        sel = bins[d][:LIB_DIGIT_SAMPLES]
        pieces_imgs.append(sel)
        pieces_labels.extend([d] * sel.shape[0])
    for op in range(4):
        pieces_imgs.append(np.stack([render_glyph(OP_GLYPHS[op]) for _ in range(LIB_OP_SAMPLES)]))
        pieces_labels.extend([10 + op] * LIB_OP_SAMPLES)
    return np.concatenate(pieces_imgs, axis=0), np.array(pieces_labels)


def per_slot_activations(enc, lib_imgs):
    """For each library image x and each slot s ∈ {a, op, b}, encode the triple
    that has x in slot s and a blank canvas in the other two; record the
    resulting trunk vector. Returns dict {slot: (N, 256)}.

    Justification: with two slots blanked, the trunk activation through unit u
    isolates the per-slot match between e(x) and the unit's prototype row
    W_u^(s), because the blank slots contribute a constant distance penalty
    that is identical across all library images.
    """
    blank = np.full((28, 28), -MEAN / STD, dtype=np.float32)
    N = lib_imgs.shape[0]
    acts = {}
    for slot_name, mask in [("a", (True, False, False)),
                            ("op", (False, True, False)),
                            ("b", (False, False, True))]:
        act = np.zeros((N, H), dtype=np.float32)
        for i in range(N):
            x = lib_imgs[i]
            a_img  = x if mask[0] else blank
            op_img = x if mask[1] else blank
            b_img  = x if mask[2] else blank
            t = enc.run(None, {
                "img_a":  a_img[None, None].astype(np.float32),
                "img_op": op_img[None, None].astype(np.float32),
                "img_b":  b_img[None, None].astype(np.float32),
            })[0][0]
            act[i] = t
        acts[slot_name] = act
    return acts


def fig_prototype_gallery(enc, dec, bins, out_path: Path, n_show: int = 16):
    print("  building library (40 digits + 16 operator glyphs)…")
    lib_imgs, lib_labels = _build_library(bins)
    print(f"  encoding library through 3 per-slot blanked feeds ({lib_imgs.shape[0]}*3 = {lib_imgs.shape[0]*3} encodes)…")
    acts = per_slot_activations(enc, lib_imgs)

    # For each unit, find the library image with max activation per slot.
    # Operator-specialised units = those whose op-slot winner is a glyph (label ≥ 10).
    win_a  = acts["a"].argmax(axis=0)
    win_op = acts["op"].argmax(axis=0)
    win_b  = acts["b"].argmax(axis=0)
    is_op_winner = lib_labels[win_op] >= 10
    n_op_specialised = int(is_op_winner.sum())
    print(f"  {n_op_specialised} / {H} units have an operator-class winner in their op slot")

    # Rank op-specialised units by op-slot activation magnitude.
    op_unit_idxs = np.where(is_op_winner)[0]
    op_unit_acts = acts["op"][win_op[op_unit_idxs], op_unit_idxs]
    top_op_units = op_unit_idxs[np.argsort(-op_unit_acts)[:n_show]]

    # 4 rows × n_show columns: row 1 = slot-a winner, row 2 = slot-op winner,
    # row 3 = slot-b winner, row 4 = caption with the (a, op, b) triple.
    fig, axes = plt.subplots(4, n_show, figsize=(1.0 * n_show, 5.0),
                             gridspec_kw={"height_ratios": [1, 1, 1, 0.35]})
    for j, u in enumerate(top_op_units):
        for r, slot, winner_idx in [(0, "a", win_a[u]),
                                    (1, "op", win_op[u]),
                                    (2, "b", win_b[u])]:
            ax = axes[r, j]
            ax.imshow(lib_imgs[winner_idx], cmap="gray", vmin=-2, vmax=2)
            ax.axis("off")
        # caption row
        lbl = lib_labels
        def _name(idx):
            return OP_GLYPHS[lbl[idx] - 10] if lbl[idx] >= 10 else str(lbl[idx])
        ax = axes[3, j]
        ax.text(0.5, 0.8, f"u {u}", ha="center", va="top", fontsize=8, family="monospace")
        ax.text(0.5, 0.25,
                f"({_name(win_a[u])}, {_name(win_op[u])}, {_name(win_b[u])})",
                ha="center", va="top", fontsize=9, color="#3A3A3A")
        ax.axis("off")
    # row labels on the leftmost column
    for r, lbl_text in [(0, "slot a"), (1, "slot op"), (2, "slot b")]:
        axes[r, 0].text(-0.18, 0.5, lbl_text, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10, family="monospace",
                        color="#3A3A3A")
    fig.suptitle(f"prototype gallery — top {n_show} operator-specialised trunk units\n"
                 f"({n_op_specialised} / {H} units total have an operator-class winner in the op slot)",
                 fontsize=11, y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("loading MNIST + ONNX sessions…")
    imgs, labs = load_mnist_test()
    bins = bin_by_digit(imgs, labs)
    enc, dec = load_sessions()

    print("collecting trunk grid (390 points)…")
    grid = collect_grid(enc, bins)

    print("generating figures…")
    fig_trunk_pca(grid, OUT / "trunk_pca_v3.png")
    fig_op_as_shift(grid, OUT / "op_as_shift.png")
    fig_decoder_unit_atlas(dec, OUT / "decoder_unit_atlas.png")
    fig_prototype_gallery(enc, dec, bins, OUT / "prototype_gallery_v3.png")
    print("done.")


if __name__ == "__main__":
    main()
