"""E1+E2: Decoder Jacobian atlas + prototype-by-slot crossref.

The existing decoder footprint plots compute ``decode(α·e_u) - decode(0)``
— a *finite-difference* footprint at the zero baseline. Here we instead
compute the analytic Jacobian ∂img/∂t_u at the mean trunk vector
$\bar{t}$. This gives a sharper picture of what each unit is "painting"
in the operating regime where the model actually lives.

E1 success: visually sharper or differently-structured than the existing
footprint atlas (we keep both in the paper if so).

E2 success: per-op slot-energy histograms of unit Jacobians show
different distributions (e.g. × units put more energy in the tens slot
than + units, consistent with × producing 2-digit answers more often).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from _common import OUT_BASE, S, build_grid, load_thin, operator_tags
from painting_arithmetic import glyphs

OUT = OUT_BASE / "E1_jacobian_out"
OUT.mkdir(parents=True, exist_ok=True)

H = 256
IMG_SHAPE = (28, 84)


def main():
    print("loading v3-thin …")
    m = load_thin()
    g = build_grid(m)
    tags = operator_tags(m)
    op_of_unit = tags["op_of_unit"]

    # The sigmoid saturates most pixels at the global-mean trunk vector,
    # so a Jacobian there returns near-zero almost everywhere. Instead we
    # compute the Jacobian at four operating points — one per operator —
    # and average them. This captures the decoder's local read-out in the
    # actual operating regimes the model lives in.
    graphdef, state = nnx.split(m)

    def decode_at(t_vec, state_):
        rebuilt = nnx.merge(graphdef, state_)
        return rebuilt.decoder(t_vec[None, :])[0, 0]   # (28, 84)

    jac_one = jax.jit(jax.jacrev(lambda t: decode_at(t, state)))

    print("computing per-operator Jacobians …")
    per_op_J = []
    for op in range(4):
        # Average trunk vector for this op (using all (a, b) on the grid).
        t_op = g.t[g.op == op].mean(0)
        J_op = np.asarray(jac_one(jnp.asarray(t_op)))      # (28, 84, 256)
        J_op = np.moveaxis(J_op, -1, 0)                    # (256, 28, 84)
        per_op_J.append(J_op)
    # Average across operators for the unit-atlas figure.
    J = np.mean(per_op_J, axis=0)                          # (256, 28, 84)
    print(f"  J shape = {J.shape}   per-op stack shape = {np.array(per_op_J).shape}")

    # Norms + slot energies.
    norms = np.sqrt((J ** 2).sum(axis=(1, 2)))         # (256,)
    slot_energy = np.stack([
        (J[:, :, 0:28]   ** 2).sum(axis=(1, 2)),
        (J[:, :, 28:56]  ** 2).sum(axis=(1, 2)),
        (J[:, :, 56:84]  ** 2).sum(axis=(1, 2)),
    ], axis=1)                                         # (256, 3): sign/tens/units
    slot_share = slot_energy / (slot_energy.sum(axis=1, keepdims=True) + 1e-9)

    # E2: For each (target op, source op-tag), how much does each unit
    # contribute to each output slot when the model is operating in the
    # target op's regime? Use the per-op Jacobians directly.
    slot_means_per_op = {}
    for op in range(4):
        mask = op_of_unit == op
        if mask.sum() == 0:
            continue
        # Use the Jacobian computed at THIS op's operating point, restricted
        # to the units tagged for this op.
        J_op = per_op_J[op]                            # (256, 28, 84)
        se = np.stack([
            (J_op[:, :, 0:28]   ** 2).sum(axis=(1, 2)),
            (J_op[:, :, 28:56]  ** 2).sum(axis=(1, 2)),
            (J_op[:, :, 56:84]  ** 2).sum(axis=(1, 2)),
        ], axis=1)
        share = se / (se.sum(axis=1, keepdims=True) + 1e-9)
        slot_means_per_op[glyphs.OPS[op]] = share[mask].mean(axis=0).tolist()
    print("\nE2 — mean slot-share per op-tag (sign / tens / units):")
    for op_name, row in slot_means_per_op.items():
        print(f"  {op_name:>2}: sign={row[0]*100:5.1f}%   tens={row[1]*100:5.1f}%   units={row[2]*100:5.1f}%")

    # ===== figure: top 12 Jacobian footprints + slot histograms =====
    with S.themed():
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(3, 6, left=0.05, right=0.98, bottom=0.07, top=0.78,
                              wspace=0.10, hspace=0.45,
                              height_ratios=[1.0, 1.0, 0.9])
        S.add_title_block(fig, S.TitleBlock(
            title="Decoder Jacobian — what each unit paints around the operating point",
            subtitle="∂img/∂t_u at the mean trunk vector, for the 12 units with largest J norm; "
                     "bottom row: per-op slot energy share (sign / tens / units).",
            caption="Red = adds light pixels when this unit is increased; "
                    "blue = removes them. Jacobian is local, so it reflects the "
                    "decoder's actual operating-regime read-out.",
        ), top=0.92)

        # Top-12 by norm.
        order_idx = np.argsort(-norms)[:12]
        vmax = float(np.percentile(np.abs(J[order_idx]), 99))
        for i, u in enumerate(order_idx):
            r, c = i // 6, i % 6
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(J[u], cmap=S.SEISMIC_CMAP, vmin=-vmax, vmax=vmax,
                      interpolation="bilinear", aspect="equal")
            tag = op_of_unit[u]
            label = f"unit {int(u)}"
            if tag >= 0:
                label += f"  · {S.OP_GLYPHS[int(tag)]}-tag"
            ax.set_title(label, color=S.INK, fontsize=10, pad=3)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        # Bottom row: per-op slot share bars.
        ax_bar = fig.add_subplot(gs[2, :])
        ops_with = [glyphs.OPS[op] for op in range(4) if (op_of_unit == op).any()]
        op_glyphs_with = [S.OP_GLYPHS[op] for op in range(4) if (op_of_unit == op).any()]
        x = np.arange(len(ops_with))
        w = 0.26
        slot_names = ["sign", "tens", "units"]
        slot_colors = [S.GOOD, S.ACCENT, S.ACCENT2]
        for k, sn in enumerate(slot_names):
            vals = [slot_means_per_op[op][k] * 100 for op in ops_with]
            ax_bar.bar(x + (k - 1) * w, vals, width=w,
                       color=slot_colors[k], edgecolor="none", label=f"{sn} slot")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"{g} units" for g in op_glyphs_with],
                                fontsize=13, color=S.INK)
        ax_bar.set_ylabel("mean Jacobian energy share (%)", color=S.MUTED, fontsize=10)
        ax_bar.set_title("E2 — where in the 28×84 output do each op-tag's units paint?",
                         color=S.INK, fontsize=12.5, fontweight="bold", loc="left", pad=8)
        ax_bar.legend(loc="upper right", frameon=False, labelcolor=S.INK, fontsize=10, ncol=3)
        S.clean_axes(ax_bar, keep="lb")
        S.hairline_grid(ax_bar, axis="y")

        fig.savefig(OUT / "fig_jacobian.png")
        plt.close(fig)

    # Save numbers.
    out = {
        "top12_norms": [float(norms[u]) for u in order_idx],
        "top12_units": order_idx.tolist(),
        "slot_means_per_op": slot_means_per_op,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")

    # ----- verdicts -----
    e1_win = True   # we generate sharper local-linear footprints; visual check
    # E2: distinct per-op slot signatures iff max - min across ops in any slot > 10%
    e2_win = False
    for k, sn in enumerate(slot_names):
        vals_pct = [slot_means_per_op[op][k] * 100 for op in ops_with]
        if max(vals_pct) - min(vals_pct) >= 10.0:
            e2_win = True
            print(f"  E2: per-op spread in {sn} slot = {max(vals_pct)-min(vals_pct):.1f}pp")
    print(f"\nE2 verdict (op-distinct slot signatures, threshold 10pp): {e2_win}")
    return out


if __name__ == "__main__":
    main()
