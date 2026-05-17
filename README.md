# Painting Arithmetic 🎨

> A tiny rational-kernel network that does single-digit MNIST arithmetic and **paints the answer as an image**. Pure latent space, JAX / Flax NNX / Grain.

```
img_a   (28×28 MNIST digit)   ─┐
img_op  (28×28 operator glyph) ─┼──► [shared encoder] ──► Yat trunk ──► [decoder]
img_b   (28×28 MNIST digit)   ─┘                                              │
                                                                              ▼
                                                                  img_out  (28×84)
                                                                  [ sign | tens | units ]
```

No softmax over result classes anywhere user-facing. The number you read is the model's last layer, pixel by pixel.

## Headline numbers (v3.1, 25 epochs)

| metric | value |
|---|---:|
| OCR test accuracy | **96.91%** |
| 14-way input-symbol recognition | 99.3% |
| Robust to operator glyph noise | 96.78% |
| Parameters | 0.81 M |
| Training time | ≈ 13 min on Apple-Silicon MPS · ≈ 3 min on a single Kaggle T4 |

Per-operator (single-digit operands, results in `[−9, 81]`):

| op | OCR | per-circle mod 2 | per-circle mod 5 | per-circle mod 11 |
|---|---:|---:|---:|---:|
| `+` | 96.2% | 97.0% | 93.0% | 93.0% |
| `−` | 96.0% | — | — | — |
| `×` | 97.1% | — | — | — |
| `÷` | 98.3% | — | — | — |

## Key interpretability findings

- **Operator regions in latent space.** PCA of the post-`h2` trunk shows the four operators occupy four distinct regions and, inside each region, the integer result moves smoothly along a direction. The trunk has learned a value line per operator.
- **Operator-as-shift.** For every pair of operators, the top singular value of the trunk-difference matrix across the 10×10 `(a, b)` grid captures **62–76 %** of the variance. The difference between any two operators is dominated by *one direction* in the 256-d trunk — a small-model reproduction of the "geometric calculator" pattern found by [Goodfire AI](https://www.goodfire.ai/research/a-geometric-calculator) inside Llama 3.1-8B.
- **Slot-localised painter units.** Pushing one-hot vectors through the trained decoder yields a per-unit "footprint" image. Many trunk units are dedicated *sign-slot painters*, *tens-slot painters*, or *units-slot painters* — the decoder has carved itself into a slot alphabet.
- **Yat prototypes are interpretable.** Each row of `h₁`'s weight matrix splits into three 64-d slots that match the three encoder embeddings. About **20 % of `h₁` units** have an operator-class prototype in their middle slot — those are the operator-conditioned arithmetic units.

The full dashboard is reproduced in [`docs/blog.md`](docs/blog.md).

## Install

```bash
git clone https://github.com/mlnomadpy/painting-arithmetic.git
cd painting-arithmetic
pip install -e .
```

## Train

```bash
# Default config (15 epochs, 30k samples per epoch). ~3 min on a Kaggle T4.
painting-arithmetic-train --epochs 15

# Reproduce the full paper number (~13 min on Apple MPS, ~5 min on a T4).
painting-arithmetic-train --epochs 25 --train-size 60000

# Toggle the encoder: --no-yat-encoder reverts to stock Conv+GELU.
painting-arithmetic-train --no-yat-encoder
```

## Evaluate a checkpoint

```bash
painting-arithmetic-eval --ckpt ./ckpts/model.npz --test-size 10000
```

## Try in a notebook

```bash
jupyter notebook notebooks/painting_arithmetic.ipynb
```

The notebook is also runnable on Kaggle — see [`notebooks/README.md`](notebooks/README.md).

## Live demo

A static, browser-side demo (three drawing pads + ONNX-Runtime-Web) lives in [`demo/`](demo/). Open `demo/index.html` after exporting a trained checkpoint to ONNX:

```bash
python scripts/export_onnx.py --ckpt ./ckpts/model.npz --out demo/model.onnx
```

See [`demo/README.md`](demo/README.md) for deployment options (GitHub Pages, Hugging Face Spaces, Cloudflare Pages).

## Architecture

The model has three pieces and several training-only auxiliary heads.

```
                    aux_sym (14-way)               training-only
                          ↑                        ┌──── mod 2 / mod 5 / mod 11 / sign
img_a   ─┐                                         │
img_op  ─┼─► SymbolEncoder ─► e_a, e_op, e_b ──► concat ──► YatNMN h₁ ──► YatNMN h₂ ──► Decoder ──► img_out
img_b   ─┘    (Yat conv or                      ℝ¹⁹²        ℝ²⁵⁶          ℝ²⁵⁶ "trunk"  ConvT      28 × 84
              stock Conv+GELU)                                   │
                                                                 └──── slot sign / tens / units
                                                                        (training-only)
```

- **Encoder.** Shared across all three input slots. Two variants — `StockEncoder` (stock `Conv + GELU`, **default and validated** at 96.91% OCR) or `YatEncoder` (every conv is a `YatConv` rational kernel; experimental, currently underperforms at the same recipe — see the model docstring). Both end with global average pool → linear → 64-d embedding.
- **Trunk.** Two `YatNMN` layers (`192 → 256 → 256`). The Yat kernel computes `α(x·W + b)² / (‖x − W‖² + ε)`, which makes each row of `W` a literal prototype point in input space. That's what makes the prototype-gallery interpretability work.
- **Decoder.** Linear projection → 16×7×21 feature map → two `ConvTranspose` upsamples → `Conv → sigmoid` → 28×84 image.
- **Auxiliary heads (training only).** A 14-way symbol classifier on each input embedding; modular CRT classifiers (mod 2, mod 5, mod 11, sign) on the trunk; per-slot classifiers (sign, tens, units) on the trunk. The per-slot CE is the load-bearing fix for the multi-digit collapse failure mode — removing it costs about 34 OCR points.

The full loss:

```
L = BCE(img_pred, target) × tens_pixel_weight    [headline, BCE with 2× weight on tens column]
  + 0.5  · CE(sign-slot, tens-slot, units-slot)  [load-bearing — the multi-digit fix]
  + 0.05 · CE(mod 2, mod 5, mod 11, sign)        [Chinese-Remainder-Theorem prior]
  + 0.2  · CE(sym_a, sym_op, sym_b)              [shared-encoder regulariser]
```

## Repository layout

```
painting-arithmetic/
├── README.md                      ← you are here
├── pyproject.toml                 ← `pip install -e .`
├── src/painting_arithmetic/
│   ├── glyphs.py                  ← font-aware symbol + result-image rendering
│   ├── data.py                    ← Grain pipeline (sampling + collation)
│   ├── model.py                   ← NNX module (encoder + trunk + decoder)
│   ├── losses.py                  ← BCE + slot CE + modular CE + symbol CE
│   ├── train.py                   ← CLI training script + library API
│   ├── eval.py                    ← OCR metric, per-op breakdown, checkpoint loader
│   └── viz.py                     ← trunk PCA, op-as-shift SVD, decoder atlas, prototype gallery
├── notebooks/
│   └── painting_arithmetic.ipynb  ← Kaggle-ready tutorial
├── docs/
│   └── blog.md                    ← long-form blog post
├── demo/
│   ├── index.html                 ← three drawing pads + live ONNX inference
│   └── README.md
├── scripts/
│   └── export_onnx.py             ← JAX → ONNX export
└── tests/
```

## Citation

If you use this work, please cite:

```bibtex
@misc{bouhsine2026painting,
  title  = {Painting Arithmetic: A Rational-Kernel Network for Visual Symbolic Computation in Latent Space},
  author = {Bouhsine, Taha},
  year   = {2026},
  url    = {https://github.com/mlnomadpy/painting-arithmetic},
}
```

The Yat rational-kernel layer comes from the [`nmn`](https://github.com/azettaai/nmn) library, which exposes it under PyTorch / Flax NNX / Flax Linen / TensorFlow / Keras / MLX backends.

## License

MIT.
