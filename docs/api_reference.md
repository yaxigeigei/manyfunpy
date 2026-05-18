# manyfunpy API Reference

`manyfunpy` is organized as a small set of focused modules. Optional configuration arguments are keyword-only where positional calls would hide intent.

## Alignment

Import from `manyfunpy.alignment`.

```python
matched_index_pairs(seq1, seq2) -> tuple[np.ndarray, np.ndarray]
align_tokens(seq1, seq2, *, gap=None) -> tuple[list[object], list[object]]
matched_times(seq1, times1, seq2, times2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Use these helpers to align token sequences with a longest-common-subsequence path. `matched_times` returns matched times from each sequence followed by the matched index arrays.

## IO

Import from `manyfunpy.io`.

```python
load_pickle(path: str | Path) -> Any
save_pickle(obj: Any, path: str | Path) -> None
```

Thin pickle helpers for notebook and analysis workflows.

## GUI

Import from `manyfunpy.gui`.

```python
create_selection_dialog(title: str, items: Sequence[T], *, multiple: bool = True) -> list[T]
```

Creates a Tk selection dialog and returns the selected items.

## Plotting

Import from `manyfunpy.mplot`.

```python
savefig(fig, out_path, *, extensions=None, is_verbose=True) -> None
paperize(h=None, *, cols_wide=None, cols_high=None, font_size=6, font_name="DejaVu Sans", zoom=2, aspect_ratio=None, journal_style="cell") -> None
plot_interval_blocks(ax, x_ranges, *, y_centers=None, heights=None, y_ranges=None, colors=None, alpha=1.0, edgecolor="none", linewidth=0, zorder=None) -> PatchCollection
get_journal_dimensions(journal_style="cell") -> dict[str, float]
axxplane(ax, coord, *, color=None, alpha=None, ylim=None, zlim=None) -> Any
axyplane(ax, coord, *, color=None, alpha=None, xlim=None, zlim=None) -> Any
axzplane(ax, coord, *, color=None, alpha=None, xlim=None, ylim=None) -> Any
axplane(ax, X_plane, Y_plane, Z_plane, *, color, alpha) -> Any
```

`paperize` formats matplotlib figures or axes for compact publication figures. `plot_interval_blocks` draws many rectangular intervals efficiently as one patch collection. The plane helpers add constant-value planes to 3D matplotlib axes.

## Data

Data helpers live under `manyfunpy.data`.

### Audio

```python
highpass_speech(waveform: np.ndarray, sample_rate: float) -> np.ndarray
compute_mel_spectrogram(tsd: nap.Tsd, source: str) -> nap.TsdFrame
estimate_sample_rate(t: np.ndarray) -> float
```

`compute_mel_spectrogram` creates an 80-band mel `TsdFrame` with source, mel-bin, and frequency metadata.

### Articulation

```python
enrich_artic(artic: nap.TsdFrame) -> nap.TsdFrame
build_artic2(frame: nap.TsdFrame) -> nap.TsdFrame
```

`enrich_artic` applies the LMV articulatory column names and adds derived constriction and derivative columns. `build_artic2` standardizes the newer articulatory secondary frame.

### Pitch

```python
enrich_pitch(pitch: nap.TsdFrame, stim_intervals: nap.IntervalSet, prod_intervals: nap.IntervalSet) -> nap.TsdFrame
bin_pitch(r_f0: np.ndarray, *, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]
```

`enrich_pitch` adds relative F0, derivatives, voicing, proxy phrase/accent tracks, and binned relative-F0 columns.

### Pynapple Containers

```python
save_nap_objects(nap_objects: Mapping[str, Any], output_dir: str | Path, *, verbose=False) -> None
warp_nap(nap_data: Mapping[str, Any], interpolant, *, sample_rate=None) -> dict[str, Any]
warp_tsd(tsd: nap.Tsd, interpolant, *, sample_rate=None) -> nap.Tsd
warp_tsdframe(tsdframe: nap.TsdFrame, interpolant, *, sample_rate=None) -> nap.TsdFrame
warp_interval_set(interval_set: nap.IntervalSet, interpolant) -> nap.IntervalSet
```

The warping helpers expect an interpolant callable that maps old timestamps to new timestamps. `warp_nap` handles dictionaries containing `Tsd`, `TsdFrame`, and `IntervalSet` values and leaves unsupported values unchanged.

### NWB Conversion

```python
convert_nwb_to_nap(nwb: str | Path | nap.NWBFile, *, ks_suffix="_KS4_Th=8") -> tuple[dict[str, Any], dict[str, dict[str, Any]]]
process_anin(nidq: nap.TsdFrame, *, mic_denoised=None) -> nap.TsdFrame
build_spike_times(nwb_nap: nap.NWBFile, ks_suffix: str, *, surface_location=None) -> nap.TsGroup
select_ks_keys(nwb_nap: nap.NWBFile, ks_suffix: str) -> list[str]
convert_to_unique_cluster_ids(cluster_ids: np.ndarray, rec_id: str, probe_index: int) -> np.ndarray
get_rec_base_num(rec_id: str) -> int
convert_cortical_depth(unit_meta, surface_location, probe_index: int) -> Any
parse_surface_location(surface_location) -> np.ndarray | float
save_nap_dataset(output_dir: str | Path, *, nap_objects=None, metadata=None) -> None
```

`convert_nwb_to_nap` returns a dictionary of pynapple objects and a dictionary of metadata tables. `ks_suffix` can be a string, a list of suffixes, or a mapping from suffix to output object name.

## Stats

Import from `manyfunpy.stats.nmf`.

```python
prepare_nonnegative_matrix(X: np.ndarray, *, neg_conversion="zero") -> tuple[np.ndarray, int]
balanced_group_sample_weights(groups: np.ndarray, *, group_totals=None) -> np.ndarray
bootstrap_gap_nmf(X: np.ndarray, *, sample_weights=None, k_list=range(1, 11), n_boot=50, fraction=0.9, n_refs=20, n_jobs=6, random_state=61, neg_conversion="zero") -> dict[str, Any]
fit_nmf_clusters(X: np.ndarray, *, sample_weights=None, n_components=None, component_n_bins=None, random_state=61, neg_conversion="zero", k_list=range(1, 11), n_boot=50, fraction=0.9, n_refs=20, n_jobs=6) -> dict[str, Any]
```

`neg_conversion` controls how signed inputs are prepared for NMF:

- `"zero"` sets negative values to zero.
- `"abs"` takes absolute values.
- `"concat"` splits positive and negative magnitudes into separate feature halves.

`fit_nmf_clusters` returns model components, signed components, projections, sorted component order, per-sample cluster IDs, and per-sample cluster weights. If `n_components` is not supplied, it also returns the `gap_summary` from `bootstrap_gap_nmf`.
