"""Export a trained JAX/NNX model to ONNX for browser inference.

Uses ``jax.numpy_dlpack`` / ``jax2tf`` → tflite → onnx pipeline. The simpler
route used here: lower the model to TensorFlow via ``jax.experimental.jax2tf``
and convert with ``tf2onnx``. Output is consumed by ``demo/index.html``
through ``onnxruntime-web``.

Usage:
    python scripts/export_onnx.py --ckpt ./ckpts/model.npz --out demo/model.onnx
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from painting_arithmetic.eval import load_checkpoint
from painting_arithmetic.model import YatArithmeticGen


def export(ckpt: str, out: str, use_yat_encoder: bool = True) -> None:
    # 1. Build + load model.
    model = YatArithmeticGen(use_yat_encoder=use_yat_encoder, rngs=nnx.Rngs(0))
    load_checkpoint(model, ckpt)

    # 2. Define a stateless inference function over (img_a, img_op, img_b).
    graphdef, state = nnx.split(model)

    def fn(state_, ia, io, ib):
        m = nnx.merge(graphdef, state_)
        return m(ia, io, ib).img

    # 3. Lower to TF SavedModel via jax2tf, then convert to ONNX with tf2onnx.
    try:
        from jax.experimental import jax2tf
        import tensorflow as tf
    except ImportError as e:
        sys.exit(
            "Need tensorflow + jax2tf installed for ONNX export.  "
            "Try `pip install tensorflow tf2onnx`.  Underlying error: " + str(e)
        )

    dummy = jnp.zeros((1, 1, 28, 28), dtype=jnp.float32)
    tf_fn = jax2tf.convert(fn, polymorphic_shapes=["...", "b, ...", "b, ...", "b, ..."])

    class Wrapper(tf.Module):
        def __init__(self):
            super().__init__()
            self.state = tf.nest.map_structure(tf.Variable, state)

        @tf.function(input_signature=[
            tf.TensorSpec((None, 1, 28, 28), tf.float32, name="img_a"),
            tf.TensorSpec((None, 1, 28, 28), tf.float32, name="img_op"),
            tf.TensorSpec((None, 1, 28, 28), tf.float32, name="img_b"),
        ])
        def __call__(self, img_a, img_op, img_b):
            return tf_fn(self.state, img_a, img_op, img_b)

    w = Wrapper()
    with tempfile.TemporaryDirectory() as td:
        sm = Path(td) / "saved_model"
        tf.saved_model.save(w, str(sm))
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", str(sm),
            "--output", out,
            "--opset", "17",
        ]
        subprocess.check_call(cmd)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="demo/model.onnx")
    p.add_argument("--no-yat-encoder", action="store_true")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    export(args.ckpt, args.out, use_yat_encoder=not args.no_yat_encoder)


if __name__ == "__main__":
    main()
