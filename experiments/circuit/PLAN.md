# Circuit-mapping appendix — plan + scorecard

We test 9 analyses on the **v3-thin** checkpoint
(`ckpts_thin/model.npz`, OCR 88.78%, single-Yat trunk h:192→256).
Each experiment has a **pre-registered success criterion**.
Only experiments that meet the bar are written up in the paper appendix.

| # | name | cost | success criterion | status | result |
|---|---|---|---|---|---|
| A3 | Prototype Gram spectrum | 0-day | effective rank meaningfully below 256; visible block structure | **WIN** | cos rank 33, op-slot rank 12, yat rank 142 / 256 |
| B1 | MI atlas (op / digits / mods / result) | 0-day | distinct vertical bands per question-class | **WIN** | 255/256 units classified into op/a/b/r slots |
| C2 | mod-k specialisation per op-tagged unit | 0-day | non-uniform distribution of (op, mod_k) preferences | weak | all ops dominated by mod-11 (result-space too small) |
| D1 | Operator embedding geometry | 0-day | 4 operators form a recognisable low-d shape; PC1 > 50% | **WIN** | PC1=84%, PC1+2=95%; − / ÷ near-collinear (cos +0.90) |
| D2 | Counterfactual steering via SVD direction | 0-day | flip rate ≥ 40% on at least one (i→j) | **WIN** | best 70% (− → ÷), three pairs > 40% |
| E1 | Decoder Jacobian atlas | 0-day | sharper / more interpretable than zero-baseline footprints | **WIN** | per-op operating-point Jacobians; sharp |
| E2 | Prototype × decoder slot crossref | 0-day | per-op slot-energy distribution differs significantly | **WIN** | × paints 75% tens; others 0–28% (74.7pp spread) |
| C1 | PySR symbolic regression on top units | tool | ≥1 unit fits closed-form R² > 0.6 at complexity ≤ 10 | fail | best R² 0.51; reported as informative negative |
| E3 | Trunk distance matrix on integer answers | 0-day | Spearman ρ between \|r−r'\| and trunk distance > 0.5 | **WIN** | ρ = 0.694 |

**Final: 6 wins + 1 informative weak negative + 1 informative fail.**
**Wrote up:** A3, B1, D1, D2, E1+E2, E3 + C1/C2 negatives. Appendix added to paper.

## Execution order

1. **A3** + **D1** — pure-geometry warm-ups, validate infra
2. **B1** + **C2** — information-theoretic atlas; sets up E2 + C1
3. **E1** + **E2** — decoder Jacobian + crossref
4. **D2** — counterfactual steering (high-stakes; either it works or we know the SVD direction is descriptive only)
5. **E3** — paint distance (expensive optimisation; only if other E results pan out)
6. **C1** — PySR (slowest; do after we know which units to target)

## Output

Each experiment writes its results + figures to
`experiments/circuit/<id>_out/`. Successful ones get an appendix
subsection in `paper/main.tex`.
