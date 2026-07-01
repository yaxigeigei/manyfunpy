import numpy as np
import pandas as pd
import pynapple as nap
from scipy.signal import savgol_filter


ARTIC_COLUMNS = [
    "tt_x",
    "tt_y",
    "td_x",
    "td_y",
    "tb_x",
    "tb_y",
    "li_x",
    "li_y",
    "ul_x",
    "ul_y",
    "ll_x",
    "ll_y",
    "la",
    "pro",
    "ttcl",
    "tbcl",
    "v_x",
    "v_y",
]

ARTIC2_COLUMNS = [
    "td_x",
    "td_y",
    "tb_x",
    "tb_y",
    "tt_x",
    "tt_y",
    "li_x",
    "li_y",
    "ul_x",
    "ul_y",
    "ll_x",
    "ll_y",
    "loudness",
    "pitch",
]


def enrich_artic(artic: nap.TsdFrame) -> nap.TsdFrame:
    """Add LMV-style articulatory track variables and derivatives."""
    # Assign column names
    df = pd.DataFrame(artic.values, index=artic.times(), columns=ARTIC_COLUMNS)

    # Compute derived variables
    df["ja"] = np.sqrt((df["li_x"] - df["ul_x"]) ** 2 + (df["li_y"] - df["ul_y"]) ** 2)
    df["ttcc"] = _constriction_cosine(df["tt_x"], df["tt_y"])
    df["tbcc"] = _constriction_cosine(df["tb_x"], df["tb_y"])
    df["tdcc"] = _constriction_cosine(df["td_x"], df["td_y"])

    # Compute derivatives
    base_cols = [col for col in df.columns if not str(col).startswith("d_")]
    for col in base_cols:
        df[f"d_{col}"] = _piecewise_derivative(df[col].to_numpy(), df.index.to_numpy())

    enriched = nap.TsdFrame(df)
    return enriched


def build_artic2(frame: nap.TsdFrame) -> nap.TsdFrame:
    """Return the direct articulatory secondary frame with standardized names."""
    df = pd.DataFrame(frame.values, index=frame.times(), columns=ARTIC2_COLUMNS)
    out = nap.TsdFrame(df)
    return out


def _constriction_cosine(x: pd.Series, y: pd.Series) -> np.ndarray:
    denom = np.sqrt(x**2 + y**2)
    denom[denom == 0] = np.nan
    return x / denom


def _piecewise_derivative(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    deriv = np.full_like(values, np.nan, dtype=np.float64)
    if len(values) < 3:
        return deriv

    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return deriv

    boundaries = np.where(np.diff(t) >= dt * 2)[0]
    starts = np.concatenate(([0], boundaries + 1))
    stops = np.concatenate((boundaries + 1, [len(values)]))
    frame_len = max(3, int(round(0.05 / dt)))
    if frame_len % 2 == 0:
        frame_len += 1

    for start, stop in zip(starts, stops, strict=False):
        seg = slice(start, stop)
        x = values[seg]
        if len(x) < 3:
            continue
        if len(x) >= frame_len:
            x = savgol_filter(x, window_length=frame_len, polyorder=2, mode="interp")
        deriv[seg] = np.gradient(x, t[seg], edge_order=1)
    return deriv
