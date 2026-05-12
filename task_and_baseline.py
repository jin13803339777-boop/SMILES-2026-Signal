"""Fixed task helpers and baseline for SMILES-2026 Signal Interference Cancellation.

This file mirrors the official starter repository. The applicant solution imports
these helpers but does not modify them.
"""

import numpy as np
from scipy.signal import convolve, firwin

CENTER = 1.9e6
BW = 0.6e6
MODEL_SUBSET = slice(20_000, 220_000)
MODEL_LAGS = tuple(range(-6, 7))
MIN_EXPLAIN_RATIO = 0.95
MAX_UNEXPLAINED_TO_RESIDUAL = 0.80


def make_bandpass(center_hz, bw_hz, fs_hz, n_taps=2047):
    lp = firwin(n_taps, bw_hz / 2, window="blackman", fs=fs_hz)
    return lp * np.exp(2j * np.pi * center_hz / fs_hz * np.arange(n_taps))


def shift_signal(x, k):
    y = np.zeros_like(x)
    if k >= 0:
        y[k:] = x[: len(x) - k]
    else:
        kk = -k
        y[: len(x) - kk] = x[kk:]
    return y


def shifted_window(x, k, start, stop):
    out = np.zeros(stop - start, dtype=np.complex128)
    src_start = max(0, start - k)
    src_stop = min(len(x), stop - k)
    if src_start >= src_stop:
        return out
    dst_start = src_start + k - start
    dst_stop = src_stop + k - start
    out[dst_start:dst_stop] = x[src_start:src_stop]
    return out


def build_task_helpers(tx_n, fs_hz, n_samples):
    score_bp = make_bandpass(CENTER, BW, fs_hz)
    locked_convolve = convolve
    locked_eigh = np.linalg.eigh
    min_explain_ratio = float(MIN_EXPLAIN_RATIO)
    max_unexplained_to_residual = float(MAX_UNEXPLAINED_TO_RESIDUAL)
    score_bp.setflags(write=False)

    def score_filter(x, kernel=score_bp, _convolve=locked_convolve):
        return _convolve(x, kernel, mode="same")

    model_terms = (
        score_filter(tx_n[:, 0] ** 2 * tx_n[:, 1].conj()),
        score_filter(tx_n[:, 1] ** 2 * tx_n[:, 0].conj()),
        score_filter(tx_n[:, 0] ** 2 * tx_n[:, 3].conj()),
        score_filter(tx_n[:, 3] ** 2 * tx_n[:, 0].conj()),
        score_filter(tx_n[:, 1] ** 2 * tx_n[:, 2].conj()),
        score_filter(tx_n[:, 2] ** 2 * tx_n[:, 1].conj()),
        score_filter(tx_n[:, 3] ** 2 * tx_n[:, 2].conj()),
        score_filter(tx_n[:, 2] ** 2 * tx_n[:, 3].conj()),
        score_filter(tx_n[:, 0] ** 2 * tx_n[:, 5].conj()),
        score_filter(tx_n[:, 5] ** 2 * tx_n[:, 0].conj()),
    )

    for term in model_terms:
        term.setflags(write=False)

    model_x = np.column_stack(
        [
            shifted_window(term, lag, MODEL_SUBSET.start, MODEL_SUBSET.stop)
            for term in model_terms
            for lag in MODEL_LAGS
        ]
    )
    model_gram = model_x.conj().T @ model_x + 1e-6 * np.eye(model_x.shape[1])
    model_x.setflags(write=False)
    model_gram.setflags(write=False)

    def apply_model_lags(term, coefs):
        pred = np.zeros(n_samples, dtype=np.complex128)
        for coef, lag in zip(coefs, MODEL_LAGS):
            pred += coef * shift_signal(term, lag)
        return pred

    def fit_tx_prediction(inp):
        pred = np.zeros_like(inp)
        for ch in range(inp.shape[1]):
            y = score_filter(inp[:, ch])[MODEL_SUBSET]
            coef = np.linalg.solve(model_gram, model_x.conj().T @ y)
            coef = coef.reshape(len(model_terms), len(MODEL_LAGS))
            ch_pred = np.zeros(n_samples, dtype=np.complex128)
            for term_idx, term in enumerate(model_terms):
                ch_pred += apply_model_lags(term, coef[term_idx])
            pred[:, ch] = ch_pred
        return pred

    def rank1_from_band_matrix(band_matrix, _eigh=locked_eigh):
        cov = band_matrix.conj().T @ band_matrix / band_matrix.shape[0]
        _, vecs = _eigh(cov)
        shared = band_matrix @ vecs[:, -1]
        denom = np.vdot(shared, shared) + 1e-30
        return np.column_stack(
            [
                (np.vdot(shared, band_matrix[:, ch]) / denom) * shared
                for ch in range(band_matrix.shape[1])
            ]
        )

    def decompose_removed_component(rx_before, rx_after):
        removed_band = np.column_stack(
            [score_filter(rx_before[:, ch] - rx_after[:, ch]) for ch in range(rx_before.shape[1])]
        )
        tx_part = fit_tx_prediction(rx_before - rx_after)
        residual = removed_band - tx_part
        rank1_part = rank1_from_band_matrix(residual)
        err = residual - rank1_part
        return removed_band, tx_part, rank1_part, err

    def explain_removed_component(rx_before, rx_after):
        removed_band, _, _, err = decompose_removed_component(rx_before, rx_after)
        total_power = np.mean(np.abs(removed_band) ** 2) + 1e-30
        err_power = np.mean(np.abs(err) ** 2)
        return 1.0 - err_power / total_power

    def score(rx_before, rx_after, label=""):
        _, _, _, err = decompose_removed_component(rx_before, rx_after)
        explain_ratio = explain_removed_component(rx_before, rx_after)
        residual_band = np.column_stack(
            [score_filter(rx_after[:, ch]) for ch in range(rx_after.shape[1])]
        )
        err_powers = np.mean(np.abs(err) ** 2, axis=0)
        residual_powers = np.mean(np.abs(residual_band) ** 2, axis=0) + 1e-30
        residual_guard = np.all(err_powers <= max_unexplained_to_residual * residual_powers)
        valid = explain_ratio >= min_explain_ratio and residual_guard

        reds = []
        if not valid:
            reasons = []
            if explain_ratio < min_explain_ratio:
                reasons.append(f"explainability {explain_ratio:.3f} < {min_explain_ratio:.2f}")
            if not residual_guard:
                worst_ratio = float(np.max(err_powers / residual_powers))
                reasons.append(
                    f"unexplained/residual {worst_ratio:.2f} > {max_unexplained_to_residual:.2f}"
                )
            print(f" INVALID: {'; '.join(reasons)}")

        for ch in range(4):
            p0 = np.mean(np.abs(score_filter(rx_before[:, ch])) ** 2)
            p1 = np.mean(np.abs(score_filter(rx_after[:, ch])) ** 2) + 1e-30
            r = 10 * np.log10(p0 / p1) if valid else 0.0
            reds.append(r)
            print(f" ch{ch}: {r:.2f} dB")
        avg = np.mean(reds)
        print(f" Metric [{label}]: {avg:.2f} dB\n")
        return reds, avg

    return {
        "score_filter": score_filter,
        "fit_tx_prediction": fit_tx_prediction,
        "explain_removed_component": explain_removed_component,
        "score": score,
    }


def baseline(tx_n, rx, fit_tx_prediction):
    """TX-only nonlinear baseline without the shared spatial residual stage."""
    del tx_n
    return rx - fit_tx_prediction(rx)
