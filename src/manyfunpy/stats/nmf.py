"""NMF clustering helpers."""

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


NegConversion = Literal["zero", "concat", "abs"]


def prepare_nonnegative_matrix(
    X: np.ndarray,
    neg_conversion: NegConversion = "zero",
) -> tuple[np.ndarray, int]:
    """Return an NMF-ready matrix, optionally concatenating negative features."""
    # Normalize values before converting signed inputs into nonnegative features.
    X = np.asarray(X, dtype=float)
    component_n_bins = X.shape[1]
    X_clean = np.nan_to_num(X, nan=0.0)

    # Convert negative values according to the requested NMF input policy.
    if neg_conversion == "concat":
        X_clean = np.column_stack((np.maximum(X_clean, 0), np.maximum(-X_clean, 0)))
    elif neg_conversion == "abs":
        X_clean = np.abs(X_clean)
    elif neg_conversion == "zero":
        X_clean = np.maximum(X_clean, 0)
    else:
        raise ValueError(f"Invalid neg_conversion: {neg_conversion}")

    return X_clean, component_n_bins


def balanced_group_sample_weights(
    groups: np.ndarray,
    group_totals: Mapping[Any, float] | None = None,
) -> np.ndarray:
    """Compute inverse group-size weights for each sample."""
    # Use observed sample counts unless external pre-filter group totals are supplied.
    groups = np.asarray(groups)

    if group_totals is None:
        unique_groups, counts = np.unique(groups, return_counts=True)
        group_totals = dict(zip(unique_groups, counts))
    else:
        group_totals = dict(group_totals)

    # Convert group totals into per-sample inverse-frequency weights.
    total = sum(group_totals.values())
    num_groups = len(group_totals)
    group_weights = {
        group: total / (num_groups * count)
        for group, count in group_totals.items()
    }

    return np.asarray([group_weights[group] for group in groups], dtype=float)


def bootstrap_gap_nmf(
    X: np.ndarray,
    sample_weights: np.ndarray | None = None,
    k_list: Sequence[int] = tuple(range(1, 11)),
    n_boot: int = 50,
    fraction: float = 0.9,
    n_refs: int = 20,
    n_jobs: int = 6,
    random_state: int = 61,
    neg_conversion: NegConversion = "zero",
) -> dict[str, Any]:
    """Run a weighted bootstrap gap-statistic sweep for pooled NMF clustering."""
    # Prepare nonnegative features and fold sample weights into the fitted matrix.
    X, component_n_bins = prepare_nonnegative_matrix(
        X,
        neg_conversion=neg_conversion,
    )
    X = np.asarray(X, dtype=float)
    if sample_weights is None:
        sample_weights = np.ones(X.shape[0], dtype=float)
    else:
        sample_weights = np.asarray(sample_weights, dtype=float)
    k_list = np.asarray(k_list, dtype=int)
    scale = np.sqrt(sample_weights)[:, None]
    X_weighted = X * scale

    def evaluate_gap(data, n_components, seed):
        # Fit NMF and assign each sample to its dominant factor.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = NMF(
                n_components=n_components,
                init="nndsvda",
                solver="cd",
                max_iter=1000,
                random_state=seed,
            )
            projection = model.fit_transform(data)

        # Compute within-cluster dispersion in the weighted feature space.
        labels = np.argmax(projection, axis=1)
        dispersion = 0.0
        for cluster_id in range(n_components):
            is_cluster = labels == cluster_id
            if not np.any(is_cluster):
                continue
            center = np.mean(data[is_cluster], axis=0)
            residual = data[is_cluster] - center
            dispersion = dispersion + np.sum(residual * residual)

        # Build reference datasets over the sampled bounding box.
        ref_rng = np.random.default_rng(seed)
        feature_min = np.min(data, axis=0)
        feature_max = np.max(data, axis=0)
        log_ref_dispersion = np.empty(n_refs, dtype=float)
        eps = np.finfo(float).tiny

        for ref_index in range(n_refs):
            ref_data = ref_rng.uniform(feature_min, feature_max, size=data.shape)
            # Fit each reference dataset using the same NMF configuration.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                ref_model = NMF(
                    n_components=n_components,
                    init="nndsvda",
                    solver="cd",
                    max_iter=1000,
                    random_state=seed + ref_index + 1,
                )
                ref_projection = ref_model.fit_transform(ref_data)

            # Compute reference dispersion using dominant-factor labels.
            ref_labels = np.argmax(ref_projection, axis=1)
            ref_dispersion = 0.0
            for cluster_id in range(n_components):
                is_cluster = ref_labels == cluster_id
                if not np.any(is_cluster):
                    continue
                center = np.mean(ref_data[is_cluster], axis=0)
                residual = ref_data[is_cluster] - center
                ref_dispersion = ref_dispersion + np.sum(residual * residual)

            log_ref_dispersion[ref_index] = np.log(max(ref_dispersion, eps))

        # Compare observed dispersion to the reference distribution.
        log_dispersion = np.log(max(dispersion, eps))
        gap_value = np.mean(log_ref_dispersion) - log_dispersion
        gap_se = np.std(log_ref_dispersion, ddof=1) * np.sqrt(1 + 1 / n_refs)
        return gap_value, gap_se, dispersion, np.mean(log_ref_dispersion)

    def evaluate_bootstrap(boot_index):
        # Evaluate the gap statistic over one subsampled bootstrap replicate.
        sample_rng = np.random.default_rng(random_state + boot_index + 1)
        sample_index = np.sort(sample_rng.choice(n_units, size=n_sample, replace=False))
        X_boot = X_weighted[sample_index]
        boot_gap_local = np.empty(len(k_list), dtype=float)
        boot_gap_se_local = np.empty(len(k_list), dtype=float)

        for k_index, n_components in enumerate(k_list):
            seed = random_state + (boot_index + 1) * 1000 + k_index
            gap_value, gap_se, _, _ = evaluate_gap(
                X_boot,
                n_components=n_components,
                seed=seed,
            )
            boot_gap_local[k_index] = gap_value
            boot_gap_se_local[k_index] = gap_se

        # Select the smallest K satisfying the one-standard-error gap rule.
        for k_index in range(len(k_list) - 1):
            if boot_gap_local[k_index] >= (
                boot_gap_local[k_index + 1] - boot_gap_se_local[k_index + 1]
            ):
                return boot_gap_local, boot_gap_se_local, int(k_list[k_index])

        return boot_gap_local, boot_gap_se_local, int(k_list[-1])

    # Run the full-data sweep and bootstrap sweep with bounded inner threads.
    n_units = X_weighted.shape[0]
    n_sample = int(np.round(n_units * fraction))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        full_results = Parallel(
            n_jobs=min(n_jobs, len(k_list)),
            verbose=10,
        )(
            delayed(evaluate_gap)(
                X_weighted,
                n_components=n_components,
                seed=random_state + k_index,
            )
            for k_index, n_components in enumerate(k_list)
        )
        boot_results = Parallel(
            n_jobs=n_jobs,
            verbose=10,
        )(
            delayed(evaluate_bootstrap)(boot_index)
            for boot_index in range(n_boot)
        )

    # Unpack the full weighted dataset sweep.
    full_gap = np.asarray([result[0] for result in full_results], dtype=float)
    full_gap_se = np.asarray([result[1] for result in full_results], dtype=float)
    full_dispersion = np.asarray([result[2] for result in full_results], dtype=float)
    full_log_ref_dispersion = np.asarray([result[3] for result in full_results], dtype=float)

    # Unpack the bootstrap K-selection sweep.
    boot_gap = np.column_stack([result[0] for result in boot_results])
    boot_gap_se = np.column_stack([result[1] for result in boot_results])
    boot_optimal_k = np.asarray([result[2] for result in boot_results], dtype=int)

    # Return the compact K-selection summary.
    summary = {
        "k_list": k_list,
        "full_gap": full_gap,
        "full_gap_se": full_gap_se,
        "full_dispersion": full_dispersion,
        "full_log_ref_dispersion": full_log_ref_dispersion,
        "boot_gap": boot_gap,
        "boot_gap_se": boot_gap_se,
        "boot_optimal_k": boot_optimal_k,
        "opti_k": int(np.median(boot_optimal_k)),
        "neg_conversion": neg_conversion,
        "component_n_bins": component_n_bins,
    }

    return summary


def fit_nmf_clusters(
    X: np.ndarray,
    sample_weights: np.ndarray | None = None,
    n_components: int | None = None,
    component_n_bins: int | None = None,
    random_state: int = 61,
    neg_conversion: NegConversion = "zero",
    k_list: Sequence[int] = tuple(range(1, 11)),
    n_boot: int = 50,
    fraction: float = 0.9,
    n_refs: int = 20,
    n_jobs: int = 6,
) -> dict[str, Any]:
    """Fit weighted pooled NMF and assign each sample to its dominant component."""
    # Prepare raw input unless the caller already supplied an NMF-ready matrix.
    if component_n_bins is None:
        X, component_n_bins = prepare_nonnegative_matrix(
            X,
            neg_conversion=neg_conversion,
        )
    else:
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        X = np.maximum(X, 0)

    # Use uniform sample weights by default.
    if sample_weights is None:
        sample_weights = np.ones(X.shape[0], dtype=float)
    else:
        sample_weights = np.asarray(sample_weights, dtype=float)

    # Search for the cluster count when no fixed component count is supplied.
    gap_summary = None
    if n_components is None:
        gap_summary = bootstrap_gap_nmf(
            X=X,
            sample_weights=sample_weights,
            k_list=k_list,
            n_boot=n_boot,
            fraction=fraction,
            n_refs=n_refs,
            n_jobs=n_jobs,
            random_state=random_state,
            neg_conversion="zero",
        )
        gap_summary["neg_conversion"] = neg_conversion
        gap_summary["component_n_bins"] = component_n_bins
        n_components = gap_summary["opti_k"]

    # Fit the final weighted NMF model.
    scale = np.sqrt(sample_weights)[:, None]
    X_weighted = X * scale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = NMF(
            n_components=n_components,
            init="nndsvda",
            solver="cd",
            max_iter=1000,
            random_state=random_state,
        )
        projection_weighted = model.fit_transform(X_weighted)
        projection = model.transform(X)

    # Sort raw components by their peak feature before any signed recombination.
    components = model.components_.copy()
    peak_index = np.argmax(components, axis=1)
    component_order = np.argsort(peak_index)
    components = components[component_order]
    projection_weighted = projection_weighted[:, component_order]
    projection = projection[:, component_order]

    # Recombine positive/negative halves only for concatenated signed inputs.
    if neg_conversion == "concat":
        signed_components = components[:, :component_n_bins] - components[:, component_n_bins:]
    else:
        signed_components = components.copy()

    # Assign each sample to its strongest sorted component.
    nmfc_id = np.argmax(projection, axis=1) + 1
    nmfc_weight = np.max(projection, axis=1)

    # Return fit artifacts used by downstream cluster plots and metadata tables.
    result = {
        "n_components": int(n_components),
        "component_n_bins": component_n_bins,
        "neg_conversion": neg_conversion,
        "components": components,
        "signed_components": signed_components,
        "component_order": component_order,
        "projection": projection,
        "projection_weighted": projection_weighted,
        "nmfc_id": nmfc_id,
        "nmfc_weight": nmfc_weight,
    }
    if gap_summary is not None:
        result["gap_summary"] = gap_summary

    return result
