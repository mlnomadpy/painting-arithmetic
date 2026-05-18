"""C1: PySR symbolic regression on operator-tagged prototype units.

For each op-tagged unit u with reasonable activation variance, fit the
function $y_u = f(a, b)$ symbolically — restricted to the canonical
$10\times 10$ $(a, b)$ grid for that operator. PySR returns a Pareto
frontier of candidate expressions trading off complexity vs. error.
We pick the simplest expression with R^2 > 0.6 (the success bar).

Success: at least one op-tagged unit's activation matches a clean
closed-form expression of $(a, b)$ with R² > 0.6 at complexity ≤ 10.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from _common import OUT_BASE, S, build_grid, load_thin, operator_tags
from painting_arithmetic import glyphs

OUT = OUT_BASE / "C1_pysr_out"
OUT.mkdir(parents=True, exist_ok=True)

# Quiet PySR.
os.environ.setdefault("JULIA_NUM_THREADS", "auto")


def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - ss_res / (ss_tot + 1e-12)


def fit_unit(a: np.ndarray, b: np.ndarray, y: np.ndarray, *,
             niterations: int = 80, time_limit_s: float = 60.0) -> dict:
    """Fit a symbolic expression y ≈ f(a, b) via PySR. Returns the simplest
    Pareto-front entry with R² > 0.6, or the best one if none reach that bar."""
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=niterations,
        populations=12,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "abs"],
        maxsize=14,
        complexity_of_constants=2,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        timeout_in_seconds=time_limit_s,
        verbosity=0,
        progress=False,
        parsimony=0.003,
        random_state=0,
        deterministic=True,
        parallelism="serial",
    )
    X = np.stack([a, b], axis=1).astype(np.float64)
    model.fit(X, y.astype(np.float64))
    eqs = model.equations_
    if eqs is None or len(eqs) == 0:
        return {"success": False, "expr": None}

    # Compute R² for each Pareto entry.
    out = []
    for _, row in eqs.iterrows():
        try:
            yhat = model.predict(X, index=row["complexity"])
        except Exception:
            continue
        out.append({
            "complexity": int(row["complexity"]),
            "loss":  float(row["loss"]),
            "r2":    float(_r2(y, yhat)),
            "expr":  str(row["equation"]),
        })

    out.sort(key=lambda d: (d["complexity"], -d["r2"]))
    # Pick the simplest with R² > 0.6.
    chosen = next((d for d in out if d["r2"] > 0.6 and d["complexity"] <= 10), None)
    if chosen is None:
        chosen = max(out, key=lambda d: d["r2"]) if out else {"success": False}
    return {
        "success": chosen.get("r2", -1) > 0.6,
        "chosen": chosen,
        "pareto": out,
    }


def main():
    print("loading v3-thin …")
    m = load_thin()
    g = build_grid(m)
    tags = operator_tags(m)
    op_of_unit = tags["op_of_unit"]
    pur_op = tags["pur_op"]

    # Pick targets: for each op, the 3 top-purity tagged units.
    targets: list[tuple[int, int]] = []
    for op in range(4):
        cand = np.where(op_of_unit == op)[0]
        if cand.size == 0:
            continue
        # Filter to units with non-trivial activation variance on that op's data.
        var_per_unit = g.t[g.op == op].var(0)
        score = pur_op[cand] + 0.1 * (var_per_unit[cand] / (var_per_unit.max() + 1e-9))
        ranked = cand[np.argsort(-score)]
        for u in ranked[:3]:
            targets.append((op, int(u)))
    print(f"selected {len(targets)} units")

    summary: list[dict] = []
    for (op, u) in targets:
        sub = g.op == op
        a, b, y = g.a[sub], g.b[sub], g.t[sub, u]
        print(f"\n=== {glyphs.OPS[op]:>2}-tag unit {u}   (n={int(sub.sum())}, y range [{y.min():.2f}, {y.max():.2f}]) ===")
        res = fit_unit(a, b, y)
        chosen = res.get("chosen", {})
        if not chosen or "expr" not in chosen:
            print("  (no fit)")
            continue
        marker = "OK" if res["success"] else "weak"
        print(f"  {marker:>4}  complexity={chosen['complexity']:>2}  R²={chosen['r2']:.3f}")
        print(f"        expr: {chosen['expr']}")
        summary.append({
            "op": glyphs.OPS[op], "unit": u, "purity": float(pur_op[u]),
            "r2": chosen["r2"], "expr": chosen["expr"],
            "complexity": chosen["complexity"], "success": res["success"],
        })

    (OUT / "results.json").write_text(json.dumps(summary, indent=2))

    win = any(d["success"] for d in summary)
    print(f"\nverdict: at least one fit with R² > 0.6? {win}")
    return summary


if __name__ == "__main__":
    main()
