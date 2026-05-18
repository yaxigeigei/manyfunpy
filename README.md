# manyfunpy

Reusable Python utilities for analysis, plotting, GUI selection, pynapple data workflows, and compact stats helpers.

## Status

Work in progress.

## Organization

- `manyfunpy.alignment`: longest-common-subsequence token alignment and matched timestamp helpers.
- `manyfunpy.io`: small pickle load/save helpers.
- `manyfunpy.gui`: Tk selection dialogs for interactive workflows.
- `manyfunpy.mplot`: matplotlib publication formatting, interval blocks, and 3D plane helpers.
- `manyfunpy.data`: pynapple-oriented audio, pitch, articulation, NWB conversion, and time-warping utilities.
- `manyfunpy.stats`: statistical helpers, currently focused on pooled NMF clustering.

## Installation

```bash
pip install -e .
```

## API Reference

Detailed function signatures and return notes live in [`docs/api_reference.md`](docs/api_reference.md).

Most optional configuration arguments are keyword-only so calls read like short analysis recipes:

```python
from manyfunpy.stats import nmf

result = nmf.fit_nmf_clusters(
    X,
    n_components=4,
    neg_conversion="concat",
    random_state=13,
)
```
