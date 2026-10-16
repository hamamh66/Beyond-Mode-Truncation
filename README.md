# Beyond Mode Truncation: Entanglement Certification from Finite Optical Apertures

Reference implementation and raw results for leakage-aware certification of spatial
entanglement from finite near- and far-field camera apertures.

The code bounds the population of a biphoton state outside a finite set of
concentration (prolate) modes from measured aperture probabilities, propagates that
bound into an outcome-resolved second-order-cone relaxation, and computes lower bounds
on the negativity and the Schmidt number of the original infinite-dimensional state.
Every reported bound is a residual-corrected dual bound with outward floating-point
error bounds. A continuum certificate encloses all errors between the continuous
optical operators and the finite computation for one geometry.

## Contents

| Path | Description |
|---|---|
| `Code/core.py` | Grid model, label-consistent aperture projectors, deficit operator with eigenvalue and Davis–Kahan margins, sources, continuum prolate eigenvalues |
| `Code/cert.py` | Measurement model; HC, GP, AG, AGr, CS, CS+AG, CS+G programs; Schmidt-number program; validated dual bounds; Hoeffding and Clopper–Pearson regions |
| `Code/continuum.py` | Continuum certificate: Nyström eigenpairs, Gauss–Legendre remainder bounds, eigenvalue enclosures, subspace and compressed-effect error bounds |
| `Code/epr.py`, `Code/E9.py` | Entropic EPR comparator |
| `Code/experiments.py` | Experiments E1–E13 |
| `Code/E14.py` | Hard-cutoff program for contamination and separable cases |
| `Code/make_figures.py` | Figures from the JSON results |
| `results/results_E*.json` | Raw outputs behind every reported number |

## Reproduction

```
pip install numpy scipy cvxpy clarabel matplotlib
cd Code
for e in E1 E2 E3 E4 E5 E6 E7 E8 E10 E11 E12 E13; do python experiments.py $e; done
python E9.py; python E14.py; python make_figures.py
```

Scripts write results to `../Supplementary/`; create that folder (or a link to
`results/`) before running. Total runtime is about 10 h on two CPU cores.
Tested with Python 3.13, NumPy, SciPy, CVXPY 1.9.3 and Clarabel.

## Citation

Please cite the accompanying article (details in `CITATION.cff`) and this archived release.

## Licence

MIT (see `LICENSE`).
