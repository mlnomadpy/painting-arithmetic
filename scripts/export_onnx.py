"""Export a trained JAX/NNX model to ONNX via ``jax2onnx`` (direct path).

Unlike :mod:`export_onnx` (which routes through ``jax2tf`` and needs
TensorFlow), this script uses :mod:`jax2onnx` to convert JAX primitives
directly to ONNX. It works on Python 3.14 where TF wheels are unavailable.

Usage::
    python scripts/export_onnx_jax2onnx.py --ckpt ./ckpts_phased_v2/model.npz --out demo/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import onnx
from jax2onnx import to_onnx

from painting_arithmetic.eval import load_checkpoint
from painting_arithmetic.model import YatArithmeticGen


def _patch_convtranspose_strides(onnx_path: str, stride: tuple[int, int] = (2, 2)) -> None:
    """Rewrite ConvTranspose strides in the exported ONNX file.

    ``jax2onnx`` (0.13.x) drops the stride attribute on ``ConvTranspose``
    lowerings, leaving it at ``[1, 1]`` and causing the decoder's two
    upsampling layers to produce 8×22 + 15×43 instead of 14×42 + 28×84.
    The JAX model is configured with ``strides=(2, 2), padding="SAME"``,
    which combined with kernel 4 and pads ``[1, 1, 1, 1]`` lowers to a
    standard ONNX ConvTranspose — only the stride is wrong.
    """
    model = onnx.load(onnx_path)
    patched = 0
    for node in model.graph.node:
        if node.op_type != "ConvTranspose":
            continue
        for attr in node.attribute:
            if attr.name == "strides":
                if list(attr.ints) != list(stride):
                    del attr.ints[:]
                    attr.ints.extend(stride)
                    patched += 1
                break
    # Strip baked-in value_info entries — they were written with the wrong
    # stride and ORT prints noisy "Falling back to lenient merge" warnings.
    del model.graph.value_info[:]
    onnx.save(model, onnx_path)
    print(f"patched {patched} ConvTranspose node(s) → strides={list(stride)}")


def export(ckpt: str, out: str, use_yat_encoder: bool = False) -> None:
    print(f"loading {ckpt} (use_yat_encoder={use_yat_encoder})")
    model = YatArithmeticGen(use_yat_encoder=use_yat_encoder, rngs=nnx.Rngs(0))
    load_checkpoint(model, ckpt)

    graphdef, state = nnx.split(model)

    # The demo only consumes the painted image. Wrap the inference call to
    # return a single tensor and stage `state` as a closed-over constant so
    # `to_onnx` only sees the three (B, 1, 28, 28) image inputs.
    def predict(ia: jnp.ndarray, io: jnp.ndarray, ib: jnp.ndarray) -> jnp.ndarray:
        m = nnx.merge(graphdef, state)
        return m(ia, io, ib).img

    inputs = [
        jax.ShapeDtypeStruct((1, 1, 28, 28), jnp.float32),
        jax.ShapeDtypeStruct((1, 1, 28, 28), jnp.float32),
        jax.ShapeDtypeStruct((1, 1, 28, 28), jnp.float32),
    ]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    to_onnx(
        predict,
        inputs,
        model_name="painting_arithmetic",
        return_mode="file",
        output_path=str(out),
        input_names=["img_a", "img_op", "img_b"],
        output_names=["img"],
    )
    print(f"wrote {out}")

    _patch_convtranspose_strides(out, stride=(2, 2))

    # Smoke check: run the exported ONNX with random inputs and compare to JAX.
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    fake = {
        "img_a": rng.standard_normal((1, 1, 28, 28)).astype(np.float32),
        "img_op": rng.standard_normal((1, 1, 28, 28)).astype(np.float32),
        "img_b": rng.standard_normal((1, 1, 28, 28)).astype(np.float32),
    }
    out_onnx = sess.run(["img"], fake)[0]
    out_jax = predict(jnp.asarray(fake["img_a"]),
                      jnp.asarray(fake["img_op"]),
                      jnp.asarray(fake["img_b"]))
    diff = np.max(np.abs(np.asarray(out_jax) - out_onnx))
    print(f"smoke check: max |jax - onnx| = {diff:.2e}  (shape {out_onnx.shape})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="demo/model.onnx")
    p.add_argument("--yat-encoder", action="store_true",
                   help="Use YatConv encoder (experimental). Default is stock Conv+GELU.")
    args = p.parse_args()
    export(args.ckpt, args.out, use_yat_encoder=args.yat_encoder)


if __name__ == "__main__":
    main()
