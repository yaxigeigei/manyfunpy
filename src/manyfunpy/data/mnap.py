import shutil
from typing import Any
from pathlib import Path


def save_nap_objects(nap_objects: dict[str, Any], output_dir: str | Path):
    """Save pynapple data to a directory."""

    if output_dir.exists():
        print(f"Removing existing pynapple data directory {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in nap_objects.items():
        value.save(output_dir / f"{key}.npz")
