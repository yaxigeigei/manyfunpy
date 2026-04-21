import shutil
import time
from typing import Any
from pathlib import Path
import numpy as np
import pynapple as nap


def save_nap_objects(nap_objects: dict[str, Any], output_dir: str | Path, verbose: bool = False):
    """Save pynapple data to a directory."""
    output_dir = Path(output_dir)

    if output_dir.exists():
        if verbose:
            print(f"Removing existing pynapple data directory {output_dir}")
        _remove_dir_with_retries(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in nap_objects.items():
        value.save(output_dir / f"{key}.npz")
        if verbose:
            print(f"Saved {output_dir / f'{key}.npz'}")

def _remove_dir_with_retries(path: Path, retries: int = 8, delay_s: float = 0.25) -> None:
    """Remove a directory, retrying around transient Windows file locks."""
    last_error = None
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay_s * (attempt + 1))
    raise PermissionError(
        f"Unable to remove {path}. It is likely open in another process "
        "(e.g., Python session, file explorer preview, or sync process). "
        f"Close handles and retry."
    ) from last_error


def warp_nap(nap_data, interpolant, sample_rate=None):
    """
    Build time-warped nap dictionary by warping all supported fields.

    Currently warps:
    - pynapple.Tsd
    - pynapple.TsdFrame
    - pynapple.IntervalSet
    """
    # Apply interpolant across data containers
    warped_data = {}
    for key, value in nap_data.items():
        if isinstance(value, nap.Tsd):
            warped_data[key] = warp_tsd(value, interpolant, sample_rate=sample_rate)
        elif isinstance(value, nap.TsdFrame):
            warped_data[key] = warp_tsdframe(value, interpolant, sample_rate=sample_rate)
        elif isinstance(value, nap.IntervalSet):
            warped_data[key] = warp_interval_set(value, interpolant)
        else:
            warped_data[key] = value
    
    return warped_data

def warp_tsd(tsd, interpolant, sample_rate=None):
    """
    Apply a time-warping interpolant to transform timestamps in a Tsd.
    """
    warped_times = interpolant(tsd.times())
    warped_support = warp_interval_set(tsd.time_support, interpolant)
    warped_tsd = nap.Tsd(
        t=warped_times,
        d=tsd.values,
        time_support=warped_support,
    )

    if sample_rate is not None:
        warped_tsd = warped_tsd.bin_average(1 / sample_rate)
        t = warped_tsd.times()
        d = warped_tsd.values.copy()
        valid = np.isfinite(d)
        if valid.any():
            d = np.interp(t, t[valid], d[valid])
        warped_tsd = nap.Tsd(
            t=t,
            d=d,
            time_support=warped_tsd.time_support,
        )

    return warped_tsd

def warp_tsdframe(tsdframe, interpolant, sample_rate=None):
    """
    Apply a time-warping interpolant to transform timestamps in a TsdFrame.
    """
    warped_times = interpolant(tsdframe.times())
    warped_support = warp_interval_set(tsdframe.time_support, interpolant)
    warped_tsdframe = nap.TsdFrame(
        t=warped_times,
        d=tsdframe.values,
        columns=tsdframe.columns,
        time_support=warped_support,
        metadata=tsdframe.metadata.copy()
    )

    if sample_rate is not None:
        warped_tsdframe = warped_tsdframe.bin_average(1 / sample_rate)
        t = warped_tsdframe.times()
        d = warped_tsdframe.values.copy()
        for i in range(d.shape[1]):
            valid = np.isfinite(d[:, i])
            if valid.any():
                d[:, i] = np.interp(t, t[valid], d[valid, i])
        warped_tsdframe = nap.TsdFrame(
            t=t,
            d=d,
            columns=warped_tsdframe.columns,
            time_support=warped_tsdframe.time_support,
            metadata=warped_tsdframe.metadata.copy(),
        )
    
    return warped_tsdframe

def warp_interval_set(interval_set, interpolant):
    """
    Applying a time-warping interpolant to transform timestamps in an IntervalSet.
    """
    starts = interpolant(np.asarray(interval_set.start, dtype=float))
    ends = interpolant(np.asarray(interval_set.end, dtype=float))
    warped_interval_set = nap.IntervalSet(
        start=starts,
        end=ends,
        metadata=interval_set.metadata.copy()
    )
    return warped_interval_set