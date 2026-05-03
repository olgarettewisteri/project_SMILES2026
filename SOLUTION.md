# Signal Interference Cancellation — Solution Report

## Reproducibility Instructions

### Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install numpy scipy gdown
```

### Running

```bash
python applicant_solution.py
```

This command will:
1. Download `challenge.mat` from Google Drive (if not already present).
2. Run the baseline and print its per-channel scores.
3. Run the proposed two-stage canceller and print its per-channel scores.
4. Write `results.json` with both sets of results.

The result of interest is `results.json["yours"]["average_db"]`.

---

## Final Solution Description

### Overview

The solution is a **two-stage interference cancellation pipeline**:

```
rx  ──[Stage 1: TX nonlinear]──▶  rx₁  ──[Stage 2: Rank-1 spatial]──▶  rx̂
```

---

### Stage 1 — Extended TX Nonlinear Canceller

**What was modified:** The baseline uses 10 IMD3 cross-product terms with lags ±6. The extended model expands both dimensions:

#### Extended cross-product terms

The signal model states the interference is a nonlinear function of TX channels jointly, specifically IMD3-type products of the form `a²·b*` and `b²·a*`. The baseline only covers a specific subset of cross-power-level pairs. The extended model adds:

| Pair type | TX columns |
|-----------|-----------|
| Within same power level | (0,1), (2,3), (4,5) |
| Cross power, same carrier | (0,2), (0,4), (1,3), (1,5), (2,4), (3,5) |
| Cross power, cross carrier | (0,3), (0,5), (1,2), (1,4), (2,5), (3,4) |

This gives **30 nonlinear basis functions** (vs 10 in the baseline).

#### Extended lag window

Lags were extended from `±6` to `±10` samples. At Fs = 7.68 MHz, ±10 samples corresponds to ±1.3 µs, which comfortably covers realistic hardware propagation delays and filter group-delay asymmetries.

#### Regularization

The baseline uses a fixed `λ = 1e-6 × I`. The extended model uses **trace-normalized Tikhonov regularization**:

```
λ = 1e-4 × trace(XᴴX) / num_features
```

This keeps the regularization scale-invariant as the number of features grows.

---

### Stage 2 — Rank-1 External Interference Cancellation

**Motivation:** The problem statement explicitly says the interference has two components:

```
I[n,c] = F_c(TX) + E[n,c]
```

where `E` is **spatially coherent** (rank-1 across 4 RX channels). After Stage 1 removes the TX-driven component, `E` remains in the residual.

**Algorithm:**

1. Compute the band-filtered residual after Stage 1: `r[n,c] = bandpass(rx₁[:,c])`.
2. Form the spatial covariance matrix `C = RᴴR / N` of shape `(4, 4)`.
3. Extract the dominant eigenvector `u` (largest eigenvalue) via `eigh`.
4. Recover the shared temporal waveform: `shared[n] = R[n,:] · u`.
5. Compute per-channel complex scaling: `αc = ⟨shared, r[:,c]⟩ / ‖shared‖²`.
6. Subtract: `rx̂[:,c] = rx₁[:,c] − αc · shared[n]`.

This is equivalent to projecting out the rank-1 component from the cross-channel covariance, which is the optimal estimator for a spatially coherent source under Gaussian assumptions.

**Validity:** The scoring checker requires the removed component to be decomposable as TX-driven + rank-1 spatial. Stage 2 directly produces a rank-1 spatial component, so validity is preserved.

---

### Why these choices?

1. **Completeness of IMD3 terms:** All 15 possible TX column pairs contribute some level of IMD3 interference. Including all of them maximizes the linear span of the regression model, at low computational cost (preprocessing is O(N) per term).

2. **Larger lag window:** Real hardware has multi-sample group delays. The baseline's ±6 may miss components at the edge; ±10 provides safety margin without overfitting risk.

3. **Stage 2 is essential:** The rank-1 external source can dominate the residual after Stage 1. PCA-based extraction is both theoretically justified and computationally cheap.

4. **Trace-normalized regularization:** The number of regression features increased 3×; naive fixed regularization would under- or over-regularize. Trace normalization ensures the penalty scales with the energy of the design matrix.

---

## Performance

| Method | Avg score |
|--------|-----------|
| No cancellation | 0.00 dB |
| Baseline (provided) | ~4 dB |
| **This solution** | **> 8 dB** (target) |

---

## Experiments and Failed Attempts

### ✗ Volterra series (5th order)

Adding 5th-order monomials (`a³·b²·c*`, etc.) was tried but discarded:
- The design matrix becomes extremely large (O(C⁵) columns).
- The additional terms showed very small regression coefficients.
- Risk of overfitting in the MODEL_SUBSET window inflating in-band score without real cancellation.

### ✗ Frequency-domain (OLS in FFT domain)

Direct frequency-domain least squares (DFT of each term, solve per-bin) was explored:
- Fast, but produced artifacts at bin boundaries when inverse-transformed.
- The time-domain formulation is more stable and sufficient.

### ✗ Adaptive filter (LMS/RLS)

Online LMS with TX nonlinear features was tried:
- Step size is hard to tune given the large dynamic range across power levels.
- RLS is numerically equivalent to batch OLS but O(N·d²) time — slower.
- Batch OLS on a representative window is preferable.

### ✗ Full-rank spatial canceller

Instead of rank-1 PCA, subtracting the full projection onto all 4 spatial modes was tried:
- This effectively zeroes out all inter-channel correlations, removing desired signal components.
- The scoring explainability check flags this as invalid (residual_guard fails).
- Rank-1 is the correct constraint from the signal model.

### ✗ Iterating Stages 1 and 2

Running Stage 1 → Stage 2 → Stage 1 again (iterative refinement) was tried:
- After one iteration, the residual in Stage 1 is already small.
- The second pass fits near-noise-floor terms and can slightly increase score but risks validity.
- Single-pass is safer and sufficiently performant.

### ✓ Kept: Trace-normalized regularization

Replacing the fixed `1e-6` regularization with `λ = 1e-4 × trace(XᴴX)/d` showed consistent improvement especially on the higher-power channels where the design matrix has larger singular values.
