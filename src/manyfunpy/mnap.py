import shutil
import time
from collections.abc import Callable, Mapping
from typing import Any
from pathlib import Path
import numpy as np
import pynapple as nap


TimeInterpolant = Callable[[np.ndarray], np.ndarray]


def save_nap_objects(
    nap_objects: Mapping[str, Any],
    output_dir: str | Path,
    verbose: bool = False,
) -> None:
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


def warp_nap(
    nap_data: Mapping[str, Any],
    interpolant: TimeInterpolant,
    sample_rate: float | None = None,
    round_decimals: int | None = 6,
) -> dict[str, Any]:
    """
    Build time-warped nap dictionary by warping all supported fields.

    Currently warps:
    - pynapple.Tsd
    - pynapple.TsdFrame
    - pynapple.IntervalSet

    Args:
        nap_data:
            Dictionary of pynapple objects and pass-through values.
        interpolant:
            Callable mapping old timestamps to warped timestamps.
        sample_rate:
            Output sampling rate in Hz for binned TsdFrame objects.
        round_decimals:
            Number of decimals for np.round on warped timestamps.

    Returns:
        Time-warped dictionary with unsupported values unchanged.
    """
    # Apply interpolant across data containers
    warped_data = {}
    for key, value in nap_data.items():
        if isinstance(value, nap.Tsd):
            warped_data[key] = warp_tsd(
                value,
                interpolant,
                round_decimals=round_decimals,
            )
        elif isinstance(value, nap.TsdFrame):
            warped_data[key] = warp_tsdframe(
                value,
                interpolant,
                sample_rate=sample_rate,
                round_decimals=round_decimals,
            )
        elif isinstance(value, nap.IntervalSet):
            warped_data[key] = warp_interval_set(
                value,
                interpolant,
                round_decimals=round_decimals,
            )
        else:
            warped_data[key] = value
    
    return warped_data

def warp_tsd(
    tsd: nap.Tsd,
    interpolant: TimeInterpolant,
    round_decimals: int | None = 6,
) -> nap.Tsd:
    """
    Apply a time-warping interpolant to transform timestamps in a Tsd.

    Args:
        tsd:
            Input one-dimensional time series.
        interpolant:
            Callable mapping old timestamps to warped timestamps.
        round_decimals:
            Number of decimals for np.round on warped timestamps.

    Returns:
        Warped Tsd with rounded timestamps.
    """
    warped_times = interpolant(tsd.times())
    if round_decimals is not None:
        warped_times = np.round(warped_times, round_decimals)
    warped_support = warp_interval_set(
        tsd.time_support,
        interpolant,
        round_decimals=round_decimals,
    )
    warped_tsd = nap.Tsd(
        t=warped_times,
        d=tsd.values,
        time_support=warped_support,
    )

    return warped_tsd

def warp_tsdframe(
    tsdframe: nap.TsdFrame,
    interpolant: TimeInterpolant,
    sample_rate: float | None = None,
    round_decimals: int | None = 6,
) -> nap.TsdFrame:
    """
    Apply a time-warping interpolant to transform timestamps in a TsdFrame.

    Args:
        tsdframe:
            Input time-by-variable frame.
        interpolant:
            Callable mapping old timestamps to warped timestamps.
        sample_rate:
            Output sampling rate in Hz after warping.
        round_decimals:
            Number of decimals for np.round on warped timestamps.

    Returns:
        Warped TsdFrame with rounded timestamps.
    """
    warped_times = interpolant(tsdframe.times())
    if round_decimals is not None:
        warped_times = np.round(warped_times, round_decimals)
    warped_support = warp_interval_set(
        tsdframe.time_support,
        interpolant,
        round_decimals=round_decimals,
    )
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
        time_support = warped_tsdframe.time_support
        if round_decimals is not None:
            t = np.round(t, round_decimals)
            time_support = nap.IntervalSet(
                start=np.round(time_support.start, round_decimals),
                end=np.round(time_support.end, round_decimals),
                metadata=time_support.metadata.copy(),
            )
        d = warped_tsdframe.values.copy()
        for i in range(d.shape[1]):
            valid = np.isfinite(d[:, i])
            if valid.any():
                d[:, i] = np.interp(t, t[valid], d[valid, i])
        warped_tsdframe = nap.TsdFrame(
            t=t,
            d=d,
            columns=warped_tsdframe.columns,
            time_support=time_support,
            metadata=warped_tsdframe.metadata.copy(),
        )
    
    return warped_tsdframe

def warp_interval_set(
    interval_set: nap.IntervalSet,
    interpolant: TimeInterpolant,
    round_decimals: int | None = 6,
) -> nap.IntervalSet:
    """
    Applying a time-warping interpolant to transform timestamps in an IntervalSet.

    Args:
        interval_set:
            Input interval table.
        interpolant:
            Callable mapping old timestamps to warped timestamps.
        round_decimals:
            Number of decimals for np.round on warped timestamps.

    Returns:
        Warped IntervalSet with rounded starts and ends.
    """
    starts = interpolant(interval_set.start)
    ends = interpolant(interval_set.end)
    if round_decimals is not None:
        starts = np.round(starts, round_decimals)
        ends = np.round(ends, round_decimals)
    warped_interval_set = nap.IntervalSet(
        start=starts,
        end=ends,
        metadata=interval_set.metadata.copy()
    )
    return warped_interval_set
