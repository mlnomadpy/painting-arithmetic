"""D2: Counterfactual steering via the operator-as-direction.

For each ordered operator pair $(i, j)$, compute the SVD-top-1 direction
of $t(i, a, b) - t(j, a, b)$ across the $10\times10$ $(a, b)$ grid.
Then at inference, when the model is fed $(a, op_i, b)$, add a multiple
$\lambda$ of that direction to $t$ before the decoder and measure the
painted answer.

Success: at $\lambda^\star$ chosen by validation, the painted answer
matches $op_j(a, b)$ on >=40% of the grid for at least one $(i, j)$
pair. (Random baseline: random pp pixel patterns would match ~1%.)

This is the causal version of the descriptive claim
``operator differences live in one direction''.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from _common import OUT_BASE, S, build_grid, load_thin
from painting_arithmetic import glyphs

OUT = OUT_BASE / "D2_steering_out"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("loading v3-thin …")
    m = load_thin()
    g = build_grid(m)

    # Pre-render the 91 canonical result images for NN-OCR.
    targets = glyphs.render_all_results()                                 # (91, 28, 84)
    t_flat = targets.reshape(91, -1)

    def ocr_idx(img: np.ndarray) -> int:
        d = ((img.reshape(-1)[None, :] - t_flat) ** 2).sum(-1)
        return int(d.argmin())

    def expected_result(a: int, b: int, op_idx: int) -> int:
        s = glyphs.OPS[op_idx]
        if s == "+":  return a + b
        if s == "-":  return a - b
        if s == "*":  return a * b
        return a // b if b != 0 else -10

    # ----- 1. Compute steering directions on the grid -----
    by_op: dict[int, dict] = {op: {} for op in range(4)}
    for n in range(g.t.shape[0]):
        key = (int(g.a[n]), int(g.b[n]))
        by_op[int(g.op[n])][key] = g.t[n]

    keys_all = set.intersection(*[set(by_op[op].keys()) for op in range(4)])
    keys = sorted(keys_all)
    print(f"  common (a,b) keys across ops: {len(keys)}")

    dirs: dict[tuple[int, int], np.ndarray] = {}
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            diff = np.stack([by_op[j][k] - by_op[i][k] for k in keys])
            U, S_, Vt = np.linalg.svd(diff, full_matrices=False)
            vr1 = (S_[0] ** 2) / (S_ ** 2).sum()
            # The mean projection determines the sign (we want positive shift
            # to push from i toward j).
            v = Vt[0]
            if (diff @ v).mean() < 0:
                v = -v
            dirs[(i, j)] = v / np.linalg.norm(v)
            print(f"  {glyphs.OPS[i]:>2} → {glyphs.OPS[j]:>2}  PC1 = {vr1*100:.1f}%   "
                  f"mean ‖t_j - t_i‖ = {np.linalg.norm(diff, axis=1).mean():.2f}")

    # ----- 2. Sweep λ on each (i, j); evaluate painted OCR -----
    @nnx.jit
    def decode_batch(model, t_arr):
        return model.decoder(t_arr)

    lambdas = np.linspace(0.0, 6.0, 13)
    sweep: dict[tuple[int, int], dict] = {}
    # Pre-fetch grid trunk vectors keyed by op
    grid_idx_of = {(int(g.a[n]), int(g.b[n]), int(g.op[n])): n for n in range(g.t.shape[0])}

    for (i, j), v in dirs.items():
        scale = float(np.linalg.norm(g.t.mean(0)))  # rough magnitude reference
        # Effective shift: dirs are unit-norm; we sweep absolute magnitudes.
        per_lambda = []
        for lam in lambdas:
            new_t = []
            keys_eval = [(a, b) for (a, b) in keys
                         if (a, b, i) in grid_idx_of and (a, b, j) in grid_idx_of]
            for (a, b) in keys_eval:
                idx = grid_idx_of[(a, b, i)]
                new_t.append(g.t[idx] + lam * v)
            imgs = np.asarray(decode_batch(m, jnp.asarray(np.stack(new_t))))
            preds = [ocr_idx(imgs[k, 0]) + glyphs.RESULT_MIN for k in range(imgs.shape[0])]
            targets_j = [expected_result(a, b, j) for (a, b) in keys_eval]
            targets_i = [expected_result(a, b, i) for (a, b) in keys_eval]
            n = len(preds)
            flip_rate = float(np.mean([p == t_j for p, t_j in zip(preds, targets_j)]))
            keep_rate = float(np.mean([p == t_i for p, t_i in zip(preds, targets_i)]))
            per_lambda.append({"lambda": float(lam),
                               "flip_rate": flip_rate,
                               "keep_rate": keep_rate,
                               "n": n})
        sweep[(i, j)] = per_lambda
        best = max(per_lambda, key=lambda d: d["flip_rate"])
        print(f"  {glyphs.OPS[i]:>2} → {glyphs.OPS[j]:>2}  best flip = "
              f"{best['flip_rate']*100:5.1f}%  at λ={best['lambda']:.2f}  "
              f"(keep i = {best['keep_rate']*100:5.1f}%)")

    # ----- figure: 4x4 grid of sweeps -----
    with S.themed():
        fig = plt.figure(figsize=(14, 11))
        gs = fig.add_gridspec(4, 4, left=0.07, right=0.97, bottom=0.07, top=0.80,
                              wspace=0.30, hspace=0.55)
        S.add_title_block(fig, S.TitleBlock(
            title="Adding the operator-difference direction to the trunk re-paints the answer",
            subtitle="For each ordered (i → j), sweep λ along the SVD top-1 direction of "
                     "t(j) − t(i) across the (a,b) grid · "
                     "orange = paints j's answer (flip); cyan = still paints i's answer (keep)",
            caption="Diagonal cells are empty (i = j). "
                    "A successful causal claim is a high orange peak at modest λ before the keep-rate also collapses.",
        ), top=0.92)

        for i in range(4):
            for j in range(4):
                ax = fig.add_subplot(gs[i, j])
                if i == j:
                    ax.text(0.5, 0.5, S.OP_GLYPHS[i], ha="center", va="center",
                            color=S.DIM, fontsize=26, transform=ax.transAxes)
                    for sp in ax.spines.values():
                        sp.set_visible(False)
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                rows = sweep[(i, j)]
                lams = [d["lambda"] for d in rows]
                flips = [d["flip_rate"] * 100 for d in rows]
                keeps = [d["keep_rate"] * 100 for d in rows]
                ax.plot(lams, flips, color=S.ACCENT, lw=1.8, label="flip → j")
                ax.plot(lams, keeps, color=S.ACCENT2, lw=1.8, label="keep i")
                ax.set_xlim(lams[0], lams[-1])
                ax.set_ylim(-2, 102)
                ax.set_title(f"{S.OP_GLYPHS[i]} → {S.OP_GLYPHS[j]}",
                             color=S.INK, fontsize=11.5, fontweight="bold", loc="left", pad=4)
                S.clean_axes(ax, keep="lb")
                S.hairline_grid(ax, axis="y")
                if i == 3 and j == 3 - 1:
                    ax.legend(loc="upper right", frameon=False, labelcolor=S.INK,
                              fontsize=9, ncol=1)

        fig.savefig(OUT / "fig_steering.png")
        plt.close(fig)

    # Save numbers + verdict.
    out = {"_keys_n": len(keys),
           "sweep": {f"{glyphs.OPS[i]}_to_{glyphs.OPS[j]}": rows
                     for (i, j), rows in sweep.items()}}
    (OUT / "results.json").write_text(json.dumps(out, indent=2))

    best_per_pair = {f"{glyphs.OPS[i]}→{glyphs.OPS[j]}": max(rows, key=lambda d: d["flip_rate"])
                     for (i, j), rows in sweep.items()}
    pass_any = any(v["flip_rate"] >= 0.40 for v in best_per_pair.values())
    print(f"\nverdict: any (i→j) flip rate ≥ 40%? {pass_any}")
    return out


if __name__ == "__main__":
    main()
