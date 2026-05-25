# Paper source

LaTeX source for *Painting Arithmetic: A Rational-Form Network for Visual Symbolic Computation in Latent Space*.

## Files

- `main.tex` — the paper itself.
- `refs.bib` — bibliography.
- `../viz_assets/` — figure PNGs (referenced via `\includegraphics{../viz_assets/...}`; assumed to live one directory up).

## Build

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## What changed in this revision

Compared to the prior draft, this revision:

- adds **Proposition 1** (Yat units are soft prototype peaks) formalising why each row of `W` acts as a prototype, after Eq. 1;
- adds **Definition 1** (rank-1 operator-as-direction hypothesis) replacing the informal SVD-variance description with a residual-budget bound `η⋆_ij ≤ 0.38`;
- adds **Definition 2** (weight-space intervention operators) casting `row`/`slot`/`random` as idempotent projections `T_X : ℝ^{256×192} → ℝ^{256×192}`;
- adds **Definition 3** (specificity score + null) replacing the informal "targeted drop minus mean off-diagonal";
- adds **Definition 4** (Theil's U) and **Definition 5** (effective rank via participation) in Appendix A;
- adds **Eq. (slot-energy)** formalising the Jacobian energy decomposition referenced in §5;
- fixes the abstract/body inconsistency on trunk depth (v3.1 = 2 Yat layers, v3-thin = 1 Yat layer; the interp results are on v3-thin);
- softens "rational kernel" to "rational form" with one sentence explaining the Mercer/RKHS disclaimer;
- new TikZ Figure 1 (architecture) on the canonical 4-color palette with `fig/` namespaced styles;
- new TikZ Figure 2 (phased curriculum) replacing the prose-only Table 1, showing trainable vs frozen modules and `stop_gradient` cuts;
- new TikZ Figure 3 (interventions as W-space projections) as the visual companion to Definitions 2–3.

## Open items

- Run the `T_random` pipeline at N = 1000 random subsets per target operator to produce a full permutation null for Definition 3. Data exists in `experiments/intervene_thin_bias_kept/`.
- `refs.bib` is a hand-typed stub; verify the entries against the canonical sources before submission.
