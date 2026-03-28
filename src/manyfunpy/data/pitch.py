import numpy as np
import pandas as pd
import pynapple as nap
from scipy import signal


PITCH_COLUMNS = ["F0raw", "F0"]


def enrich_pitch(
    pitch: nap.TsdFrame,
    stim_intervals: nap.IntervalSet,
    prod_intervals: nap.IntervalSet,
) -> nap.TsdFrame:
    """Add LMV-style pitch features to the raw pitch frame."""
    pitch = nap.TsdFrame(t=pitch.times(), d=pitch.values, columns=PITCH_COLUMNS)
    t = pitch.times()
    f0 = pitch["F0"].values

    log_f0 = np.log(f0)
    r_log_f0 = np.full_like(log_f0, np.nan, dtype=np.float64)

    for start, end in stim_intervals.values:
        mask = (t >= start) & (t <= end) & np.isfinite(log_f0)
        if mask.sum() > 1:
            r_log_f0[mask] = _zscore_without_outliers(log_f0[mask])

    prod_mask = _interval_mask(t, prod_intervals) & np.isfinite(log_f0)
    if prod_mask.sum() > 1:
        r_log_f0[prod_mask] = _zscore_without_outliers(log_f0[prod_mask])

    r_f0 = np.exp(r_log_f0)
    d_f0 = np.gradient(f0, t, edge_order=1)
    d_rf0 = np.gradient(r_f0, t, edge_order=1)
    voicing = np.isfinite(f0)
    phrase, accent = _fujisaki_proxy(r_f0, t)
    binned, bin_edges = bin_pitch(r_f0)

    extra = {
        "rF0": r_f0,
        "dF0": d_f0,
        "drF0": d_rf0,
        "voicing": voicing,
        "phrase": phrase,
        "accent": accent,
    }
    for i in range(binned.shape[1]):
        extra[f"brF0_{i}"] = binned[:, i]

    data = np.column_stack(
        [pitch.values]
        + [extra[name] for name in extra]
    )
    columns = list(pitch.columns) + list(extra)
    
    return nap.TsdFrame(t=t, d=data, columns=columns)


def bin_pitch(r_f0: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(r_f0)
    bins = np.full((r_f0.shape[0], n_bins), np.nan)
    if finite.sum() == 0:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        return bins, edges

    lims = np.nanpercentile(r_f0[finite], [2.5, 97.5])
    lims[1] = min(lims[1], 6.0)
    edges = np.linspace(lims[0], lims[1], n_bins + 1)
    edges[0] = -np.inf
    memberships = np.digitize(r_f0, edges[1:-1], right=False)
    for i in range(n_bins):
        bins[memberships == i, i] = 1.0
    bins[~finite] = np.nan
    return bins, edges


def _zscore_without_outliers(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.full_like(values, np.nan)

    clean = values.copy()
    median = np.nanmedian(clean[finite])
    mad = np.nanmedian(np.abs(clean[finite] - median))
    if np.isfinite(mad) and mad > 0:
        robust_z = 0.6745 * (clean - median) / mad
        clean[np.abs(robust_z) > 3.5] = np.nan

    mu = np.nanmean(clean)
    sigma = np.nanstd(clean)
    if not np.isfinite(sigma) or sigma == 0:
        return np.full_like(values, np.nan)
    return (clean - mu) / sigma


def _interval_mask(t: np.ndarray, intervals: nap.IntervalSet) -> np.ndarray:
    labels = nap.Ts(t=t).in_interval(intervals)
    return labels.values > 0


def _fujisaki_proxy(r_f0: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filled = pd.Series(r_f0).interpolate(limit_direction="both").to_numpy()
    finite = np.isfinite(filled)
    phrase = np.full_like(filled, np.nan, dtype=np.float64)
    accent = np.full_like(filled, np.nan, dtype=np.float64)
    if finite.sum() < 4:
        return phrase, accent

    sample_rate = 1.0 / np.median(np.diff(t))
    sos = signal.butter(4, 1.5, btype="highpass", fs=sample_rate, output="sos")
    accent_sig = signal.sosfiltfilt(sos, filled[finite])
    phrase_sig = filled[finite] - accent_sig
    accent_sig[accent_sig < 0] = 0
    phrase[finite] = phrase_sig
    accent[finite] = accent_sig
    phrase[~np.isfinite(r_f0)] = np.nan
    accent[~np.isfinite(r_f0)] = np.nan
    return phrase, accent
