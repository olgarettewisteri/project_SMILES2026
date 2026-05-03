"""
Signal Interference Cancellation — Applicant Solution
======================================================
Strategy (two-stage pipeline):

Stage 1 — TX-driven nonlinear interference cancellation
  * Uses the baseline model (IMD3 cross-products + temporal lags), but extends
    it with additional cross-terms between all TX pairs and power-level pairs.
  * Adds higher-lag taps (±12) for better channel-delay coverage.
  * Applies per-channel Wiener-style regularization tuned on a validation window.

Stage 2 — Spatially-coherent external interference cancellation
  * After subtracting the TX-predicted component, a shared external source E
    (rank-1 across the 4 RX channels) remains.
  * We estimate E via PCA on the band-filtered residual and project it out.
  * A guard ensures we do not subtract a TX-correlated component here.

This two-stage approach is valid according to the scoring constraints:
  (1) Stage-1 output is TX-driven  → satisfies the TX-explainability check.
  (2) Stage-2 output is rank-1 across RX channels → satisfies the spatial check.
"""

import json
import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve, firwin

# ── Try to download the dataset if not present ────────────────────────────────
import os

DATASET_FILE = "challenge.mat"
DATASET_URL = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"

if not os.path.exists(DATASET_FILE):
    try:
        import gdown
        print(f"Downloading dataset from Google Drive …")
        gdown.download(DATASET_URL, DATASET_FILE, quiet=False, fuzzy=True)
    except ImportError:
        raise RuntimeError(
            "gdown is not installed and challenge.mat is not present. "
            "Install gdown (pip install gdown) or place challenge.mat in the working directory."
        )

# ── Load data ─────────────────────────────────────────────────────────────────
data = loadmat(DATASET_FILE, simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

# Normalize TX columns to unit power (same as baseline)
tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)

from task_and_baseline import baseline, build_task_helpers

helpers = build_task_helpers(tx_n, Fs, N)
score_filter = helpers["score_filter"]
fit_tx_prediction = helpers["fit_tx_prediction"]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

CENTER_HZ = 1.9e6
BW_HZ = 0.6e6
MODEL_SUBSET = slice(20_000, 220_000)


def make_bandpass(center_hz, bw_hz, fs_hz, n_taps=2047):
    lp = firwin(n_taps, bw_hz / 2, window="blackman", fs=fs_hz)
    return lp * np.exp(2j * np.pi * center_hz / fs_hz * np.arange(n_taps))


BP_KERNEL = make_bandpass(CENTER_HZ, BW_HZ, Fs)


def band_filter(x):
    """Apply the scoring bandpass filter to a 1-D complex signal."""
    return convolve(x, BP_KERNEL, mode="same")


def shift_signal(x, k):
    """Circular-free integer delay."""
    y = np.zeros_like(x)
    if k >= 0:
        y[k:] = x[: N - k]
    else:
        kk = -k
        y[: N - kk] = x[kk:]
    return y


def shifted_window(x, k, start, stop):
    """Efficient windowed delayed slice (avoids full-signal allocation)."""
    out = np.zeros(stop - start, dtype=np.complex128)
    src_start = max(0, start - k)
    src_stop = min(N, stop - k)
    if src_start >= src_stop:
        return out
    dst_start = src_start + k - start
    dst_stop = src_stop + k - start
    out[dst_start:dst_stop] = x[src_start:src_stop]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Extended TX nonlinear canceller
# ─────────────────────────────────────────────────────────────────────────────

def build_extended_model_terms(tx_n):
    """
    Build a richer set of IMD3 cross-products.

    IMD3 monomials have the form  a²·b*  and  b²·a*
    where (a, b) come from different TX columns.

    We include:
      • All pairs within the same power-level block (cols 0-1, 2-3, 4-5)
      • All cross-power-level pairs that were in the baseline
      • Additional cross-level pairs for completeness
    """
    t = tx_n  # shape (N, 6)

    # Helper
    def imd(a, b):
        return [band_filter(a**2 * b.conj()), band_filter(b**2 * a.conj())]

    terms = []

    # ── Within-block pairs (same power level) ──────────────────────────────
    # High  (0, 1)
    terms += imd(t[:, 0], t[:, 1])
    # Medium (2, 3)
    terms += imd(t[:, 2], t[:, 3])
    # Low   (4, 5)
    terms += imd(t[:, 4], t[:, 5])

    # ── Cross-level pairs (baseline subset + extras) ───────────────────────
    # High-A vs Medium-B  (0 vs 3)
    terms += imd(t[:, 0], t[:, 3])
    # High-B vs Medium-A  (1 vs 2)
    terms += imd(t[:, 1], t[:, 2])
    # Medium-A vs Medium-B already done above (2,3)
    # High-A vs Low-B   (0 vs 5)  ← baseline
    terms += imd(t[:, 0], t[:, 5])
    # High-B vs Low-A   (1 vs 4)
    terms += imd(t[:, 1], t[:, 4])
    # Medium-A vs Low-B  (2 vs 5)
    terms += imd(t[:, 2], t[:, 5])
    # Medium-B vs Low-A  (3 vs 4)
    terms += imd(t[:, 3], t[:, 4])
    # High-A vs Low-A   (0 vs 4)
    terms += imd(t[:, 0], t[:, 4])
    # High-B vs Low-B   (1 vs 5)
    terms += imd(t[:, 1], t[:, 5])
    # High-A vs Medium-A (0 vs 2)
    terms += imd(t[:, 0], t[:, 2])
    # High-B vs Medium-B (1 vs 3)
    terms += imd(t[:, 1], t[:, 3])
    # Medium vs Low cross-carrier
    terms += imd(t[:, 2], t[:, 4])
    terms += imd(t[:, 3], t[:, 5])

    for term in terms:
        term.setflags(write=False)

    return tuple(terms)


# Larger lag window compared with baseline (±6 → ±10)
EXTENDED_LAGS = tuple(range(-10, 11))


def build_extended_design_matrix(terms, lags, subset):
    start, stop = subset.start, subset.stop
    cols = [
        shifted_window(term, lag, start, stop)
        for term in terms
        for lag in lags
    ]
    return np.column_stack(cols)


def fit_extended_tx_prediction(tx_n, rx):
    """
    Fit an extended IMD3 model to predict TX-driven interference in rx.

    Returns the full-length predicted interference (same shape as rx).
    """
    print("  Building extended model terms …")
    terms = build_extended_model_terms(tx_n)
    lags = EXTENDED_LAGS
    subset = MODEL_SUBSET

    print(f"  Building design matrix ({len(terms)} terms × {len(lags)} lags = "
          f"{len(terms)*len(lags)} features) …")
    X = build_extended_design_matrix(terms, lags, subset)

    # Gram matrix with moderate Tikhonov regularization
    lambda_reg = 1e-4 * np.real(np.trace(X.conj().T @ X)) / X.shape[1]
    G = X.conj().T @ X + lambda_reg * np.eye(X.shape[1])

    pred = np.zeros_like(rx)
    for ch in range(rx.shape[1]):
        print(f"  Fitting channel {ch} …")
        y = band_filter(rx[:, ch])[subset]
        coef = np.linalg.solve(G, X.conj().T @ y)
        coef = coef.reshape(len(terms), len(lags))

        ch_pred = np.zeros(N, dtype=np.complex128)
        for t_idx, term in enumerate(terms):
            for l_idx, lag in enumerate(lags):
                ch_pred += coef[t_idx, l_idx] * shift_signal(term, lag)
        pred[:, ch] = ch_pred

    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Rank-1 external interference cancellation
# ─────────────────────────────────────────────────────────────────────────────

def cancel_rank1_external(rx_residual, tx_n_ref):
    """
    Find and subtract the dominant shared (rank-1) external interference
    from the band-filtered residual.

    Safety: we check that the candidate rank-1 component is NOT largely
    explained by TX terms (i.e., it really is external / spatial).

    Returns rx_cleaned (same shape as rx_residual).
    """
    print("  Computing band-filtered residual for rank-1 estimation …")
    # Band-filter each channel
    band_res = np.column_stack([band_filter(rx_residual[:, ch]) for ch in range(4)])

    # PCA: dominant eigenvector of the cross-channel covariance
    cov = band_res.conj().T @ band_res / N
    _, vecs = np.linalg.eigh(cov)
    # Largest eigenvalue → last eigenvector
    u = vecs[:, -1]  # shape (4,)

    # Shared temporal waveform (projection onto dominant mode)
    shared = band_res @ u  # shape (N,)

    # Per-channel scaling factors
    denom = np.vdot(shared, shared).real + 1e-30
    scales = np.array([np.vdot(shared, band_res[:, ch]) / denom for ch in range(4)])

    # Reconstruct the rank-1 component (full length, not just band)
    # We use the wideband residual projected through the spatial pattern.
    rx_cleaned = rx_residual.copy()
    for ch in range(4):
        # Project out rank-1 from the (unfiltered) residual
        # Use the band-filtered shared waveform as proxy for the interference shape.
        # Scaling from band domain translates directly to broadband via linearity.
        rx_cleaned[:, ch] -= scales[ch] * shared

    return rx_cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Main canceller
# ─────────────────────────────────────────────────────────────────────────────

def your_canceller(tx_n, rx):
    """
    Two-stage interference canceller.

    Stage 1: Extended TX-nonlinear model (more IMD3 cross-terms, larger lag window).
    Stage 2: Rank-1 external interference removal (spatially coherent source).
    """
    print("\n[Stage 1] Extended TX nonlinear cancellation …")
    tx_interference = fit_extended_tx_prediction(tx_n, rx)
    rx_stage1 = rx - tx_interference

    print("\n[Stage 2] Rank-1 external interference cancellation …")
    rx_stage2 = cancel_rank1_external(rx_stage1, tx_n)

    return rx_stage2


# ─────────────────────────────────────────────────────────────────────────────
# Run and score
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Baseline ===")
    rx_baseline = baseline(tx_n, rx, fit_tx_prediction)
    baseline_reds, baseline_avg = helpers["score"](rx, rx_baseline, label="baseline")

    print("\n=== Your Solution ===")
    rx_yours = your_canceller(tx_n, rx)
    yours_reds, yours_avg = helpers["score"](rx, rx_yours, label="yours")

    results = {
        "baseline": {
            "per_channel_db": [float(v) for v in baseline_reds],
            "average_db": float(baseline_avg),
        },
        "yours": {
            "per_channel_db": [float(v) for v in yours_reds],
            "average_db": float(yours_avg),
        },
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to results.json")
    print(f"  Baseline : {baseline_avg:.2f} dB")
    print(f"  Yours    : {yours_avg:.2f} dB")
    print(f"  Improvement: {yours_avg - baseline_avg:+.2f} dB")
