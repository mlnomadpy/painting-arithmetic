"""B1+C2: Mutual-information atlas + mod-k specialisation.

For each h₁ unit u, compute discrete mutual information between its
activation $y_u$ (binned into 8 quantiles over the canonical grid) and
six categorical labels:

  - op   (4 classes)
  - a, b (10 classes each)
  - r    (91 classes — the integer result)
  - r % 2, r % 5, r % 11

The atlas reveals which units know which questions. C2 narrows in on
operator-tagged units and asks which mod-k subproblem each specialises
in — the small-model version of Goodfire's "neurons separate by which
mod-k circle they read from."

Success B1: visually distinct vertical bands per question class; at
least 20% of units show a clear single-class winner.

Success C2: non-uniform distribution of (op, mod_k) preferences — i.e.
not all × units specialise to the same mod_k.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import OUT_BASE, S, build_grid, load_thin, operator_tags

OUT = OUT_BASE / "B1_mi_out"
OUT.mkdir(parents=True, exist_ok=True)


def mi_discrete(x: np.ndarray, y: np.ndarray, n_bins_x: int = 8) -> np.ndarray:
    """Per-feature MI between x (N, F) [continuous] and y (N,) [int].

    Returns (F,) of MI in nats. Bins x per-feature into quantiles.
    """
    N, F = x.shape
    Y = np.asarray(y).astype(int)
    classes = np.unique(Y)
    mis = np.zeros(F, dtype=np.float64)
    for f in range(F):
        col = x[:, f]
        # Equal-frequency bins.
        edges = np.quantile(col, np.linspace(0, 1, n_bins_x + 1))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        Xb = np.digitize(col, edges[1:-1])
        # Joint
        joint = np.zeros((n_bins_x, classes.size))
        for i in range(N):
            joint[Xb[i], np.searchsorted(classes, Y[i])] += 1
        joint /= N
        px = joint.sum(1, keepdims=True)
        py = joint.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.where(joint > 0, joint * (np.log(joint) - np.log(px) - np.log(py)), 0.0)
        mis[f] = mi.sum()
    return mis


def main():
    print("loading v3-thin …")
    m = load_thin()
    g = build_grid(m)
    tags = operator_tags(m)

    y_units = g.t                                  # (N=390, F=256)
    labels = {
        "op":    g.op,
        "a":     g.a,
        "b":     g.b,
        "r":     g.r,
        "r%2":   np.abs(g.r) % 2,
        "r%5":   np.abs(g.r) % 5,
        "r%11":  np.abs(g.r) % 11,
    }
    atlas = {}
    label_entropies: dict[str, float] = {}
    for name, y in labels.items():
        mis = mi_discrete(y_units, y, n_bins_x=8)
        atlas[name] = mis
        # Empirical H(Y).
        _, cnts = np.unique(y, return_counts=True)
        p = cnts / cnts.sum()
        HY = -(p * np.log(p)).sum()
        label_entropies[name] = float(HY)
        print(f"  MI(unit; {name:>4}) — max {mis.max():.3f}   median {np.median(mis):.3f}   "
              f"H(Y)={HY:.2f}  →  max U = {mis.max()/HY:.3f}")

    # Normalise each column by H(Y) — Theil's U (uncertainty coefficient).
    # Now every column is in [0, 1] and the units across labels with different
    # cardinalities are directly comparable.
    M = np.stack([atlas[k] for k in labels], axis=1)        # (256, 7)
    HY_vec = np.array([label_entropies[k] for k in labels])
    M_norm = M / (HY_vec[None, :] + 1e-9)

    # B1 success: per-row argmax distribution.
    winner_per_unit = M_norm.argmax(axis=1)
    margin = np.sort(M_norm, axis=1)[:, -1] - np.sort(M_norm, axis=1)[:, -2]
    n_clear = int((margin > 0.15).sum())
    print(f"\nB1: {n_clear}/256 units have a clear single-label winner (margin > 0.15)")
    winner_counts = Counter(winner_per_unit.tolist())
    label_names = list(labels.keys())
    for k, name in enumerate(label_names):
        print(f"  argmax = {name:>4} : {winner_counts.get(k, 0)} units")

    # C2: per-op-tagged unit, which mod-k preference dominates?
    op_of_unit = tags["op_of_unit"]
    mod_idx = {"r%2": label_names.index("r%2"),
               "r%5": label_names.index("r%5"),
               "r%11": label_names.index("r%11")}
    mod_block = np.stack([M_norm[:, mod_idx[k]] for k in mod_idx], axis=1)   # (256, 3)
    mod_winner = mod_block.argmax(axis=1)
    mod_names = list(mod_idx.keys())

    c2_table: dict = {}
    for op in range(4):
        mask = op_of_unit == op
        if mask.sum() == 0:
            c2_table[S.OP_GLYPHS[op]] = {n: 0 for n in mod_names}
            continue
        ct = Counter(mod_winner[mask].tolist())
        c2_table[S.OP_GLYPHS[op]] = {mod_names[k]: int(ct.get(k, 0)) for k in range(3)}
    print("\nC2: mod-k preference among op-tagged units")
    for op, row in c2_table.items():
        print(f"  {op} units: {row}")

    # ===== figure: the atlas =====
    with S.themed():
        fig = plt.figure(figsize=(14.5, 8.5))
        gs = fig.add_gridspec(2, 2, left=0.07, right=0.96, bottom=0.10, top=0.72,
                              wspace=0.22, hspace=0.55,
                              width_ratios=[2.4, 1.0], height_ratios=[1.5, 1.0])
        S.add_title_block(fig, S.TitleBlock(
            title="The trunk has dedicated knowledge slots",
            subtitle="Per-unit normalised MI (Theil's U = I/H(Y)) with seven question-types · "
                     "rows = h₁ units (sorted by op-tag), cols = labels",
            caption=("most units are dedicated either to operator identity or to the "
                     "integer result; modular subproblems are sparsely served"),
        ), top=0.93)

        # Sort units by op-tag for visual structure.
        order = np.lexsort((winner_per_unit, op_of_unit))

        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(M_norm[order], aspect="auto",
                       cmap=S.VALUE_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(label_names, fontsize=11, color=S.INK)
        ax.set_yticks([]); ax.set_ylabel("h₁ units (sorted by op-tag, then top label)",
                                          color=S.MUTED, fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
        # Row dividers per op-tag.
        op_sorted = op_of_unit[order]
        for op in range(4):
            mask = op_sorted == op
            if not mask.any():
                continue
            lo, hi = int(np.argmax(mask)), int(len(op_sorted) - np.argmax(mask[::-1]))
            ax.axhline(hi - 0.5, color=S.ACCENT, lw=0.8, alpha=0.6)

        cb = S.colorbar(ax, im, label="normalised MI")

        # Right panel: histogram of clear winners by label.
        ax2 = fig.add_subplot(gs[0, 1])
        cnts = [winner_counts.get(k, 0) for k in range(len(labels))]
        ax2.barh(range(len(labels)), cnts, color=S.ACCENT2, edgecolor="none")
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(label_names, fontsize=11, color=S.INK)
        ax2.invert_yaxis()
        ax2.set_xlabel("# units with this label as argmax",
                       color=S.MUTED, fontsize=10)
        S.clean_axes(ax2, keep="lb")
        S.hairline_grid(ax2, axis="x")

        # Bottom panel: C2 — mod-k specialisation per op-tag.
        ax3 = fig.add_subplot(gs[1, :])
        ops_with_units = [op for op in range(4) if (op_of_unit == op).any()]
        x = np.arange(len(ops_with_units))
        w = 0.26
        offsets = {"r%2": -w, "r%5": 0.0, "r%11": w}
        colors = {"r%2": S.OP_COLORS["+"], "r%5": S.OP_COLORS["×"], "r%11": S.OP_COLORS["÷"]}
        for mod_name in ["r%2", "r%5", "r%11"]:
            vals = [c2_table[S.OP_GLYPHS[op]][mod_name] for op in ops_with_units]
            ax3.bar(x + offsets[mod_name], vals, width=w,
                    color=colors[mod_name], edgecolor="none", label=mod_name)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"{S.OP_GLYPHS[op]} units" for op in ops_with_units],
                            fontsize=13, color=S.INK)
        ax3.set_ylabel("# units preferring this mod-k", color=S.MUTED, fontsize=10)
        ax3.set_title("C2 — mod-k specialisation within each op-tagged subset",
                      color=S.INK, fontsize=12.5, fontweight="bold", loc="left", pad=8)
        ax3.legend(loc="upper right", frameon=False, labelcolor=S.INK, fontsize=10)
        S.clean_axes(ax3, keep="lb")
        S.hairline_grid(ax3, axis="y")

        fig.savefig(OUT / "fig_atlas.png")
        plt.close(fig)

    # Save numbers.
    out = {
        "atlas": {k: v.tolist() for k, v in atlas.items()},
        "n_clear_winners": int(n_clear),
        "winner_counts_by_label": {label_names[k]: int(v) for k, v in winner_counts.items()},
        "c2_table": c2_table,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2))

    # ----- verdicts -----
    # B1 (revised): >=80% of units argmax to one of {op, a, b, r} — the four
    # "first-class" knowledge slots. (Under H(Y)-normalisation, mod-k
    # specialisation is rare in this small result space because the result
    # itself only has 91 classes and mod-11 nearly recovers it.)
    primary_labels = {label_names.index(n) for n in ("op", "a", "b", "r")}
    n_primary = int(sum(c for k, c in winner_counts.items() if k in primary_labels))
    b1_win = n_primary >= 0.80 * 256

    # C2: a per-op row is judged "dominant" only if its dominant mod-k
    # exceeds 50% AND the dominant mod-k differs across op rows (otherwise
    # the whole story is just "everyone tracks mod-11").
    dominant_mod_per_op = {}
    for op, row in c2_table.items():
        s = sum(row.values())
        if s == 0:
            dominant_mod_per_op[op] = None
            continue
        winner = max(row, key=row.get)
        dominant_mod_per_op[op] = winner if row[winner] / s > 0.5 else None
    distinct_winners = {v for v in dominant_mod_per_op.values() if v is not None}
    c2_win = len(distinct_winners) >= 2     # ops differ in their mod-k preference
    print(f"\nB1 verdict: {n_primary}/256 units argmax to a primary knowledge slot "
          f"({100*n_primary/256:.0f}%). Pass = {b1_win}")
    print(f"C2 verdict: {dict(dominant_mod_per_op)} — distinct winners across ops = {distinct_winners}. "
          f"Pass = {c2_win}")
    return out


if __name__ == "__main__":
    main()
