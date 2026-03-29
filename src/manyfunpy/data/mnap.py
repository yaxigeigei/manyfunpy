import shutil
import time
from typing import Any
from pathlib import Path


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


def save_nap_objects(nap_objects: dict[str, Any], output_dir: str | Path):
    """Save pynapple data to a directory."""
    output_dir = Path(output_dir)

    if output_dir.exists():
        print(f"Removing existing pynapple data directory {output_dir}")
        _remove_dir_with_retries(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in nap_objects.items():
        value.save(output_dir / f"{key}.npz")
