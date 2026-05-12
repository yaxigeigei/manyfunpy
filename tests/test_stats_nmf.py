import numpy as np

from manyfunpy.stats import nmf


def test_prepare_nonnegative_matrix_zeroes_negative_input_by_default():
    X = np.array([[1.0, -2.0, np.nan], [-3.0, 4.0, 0.0]])

    X_nmf, component_n_bins = nmf.prepare_nonnegative_matrix(X)

    assert component_n_bins == 3
    np.testing.assert_allclose(
        X_nmf,
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ]),
    )


def test_prepare_nonnegative_matrix_splits_all_nonnegative_input_when_concat():
    X = np.array([[1.0, 2.0], [np.nan, 4.0]])

    X_nmf, component_n_bins = nmf.prepare_nonnegative_matrix(
        X,
        neg_conversion="concat",
    )

    assert component_n_bins == 2
    np.testing.assert_allclose(
        X_nmf,
        np.array([
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
        ]),
    )


def test_prepare_nonnegative_matrix_zeroes_negative_input_explicitly():
    X = np.array([[1.0, -2.0, np.nan], [-3.0, 4.0, 0.0]])

    X_nmf, component_n_bins = nmf.prepare_nonnegative_matrix(
        X,
        neg_conversion="zero",
    )

    assert component_n_bins == 3
    np.testing.assert_allclose(
        X_nmf,
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ]),
    )


def test_balanced_group_sample_weights():
    groups = np.array(["a", "a", "b", "c", "c", "c"])

    sample_weights = nmf.balanced_group_sample_weights(groups)

    np.testing.assert_allclose(
        sample_weights,
        np.array([1.0, 1.0, 2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]),
    )


def test_fit_nmf_clusters_shapes_and_component_sorting():
    X_raw = np.array([
        [4.0, 1.0, -0.1, 0.0],
        [3.5, 0.8, -0.2, 0.0],
        [0.0, -0.2, 1.0, 4.0],
        [0.0, -0.1, 0.8, 3.5],
    ])
    X_nmf, component_n_bins = nmf.prepare_nonnegative_matrix(
        X_raw,
        neg_conversion="concat",
    )

    result = nmf.fit_nmf_clusters(
        X=X_nmf,
        sample_weights=np.ones(X_nmf.shape[0]),
        n_components=2,
        component_n_bins=component_n_bins,
        neg_conversion="concat",
        random_state=13,
    )

    assert result["components"].shape == (2, X_nmf.shape[1])
    assert result["signed_components"].shape == (2, component_n_bins)
    assert result["projection"].shape == (X_nmf.shape[0], 2)
    assert result["projection_weighted"].shape == (X_nmf.shape[0], 2)
    assert result["nmfc_id"].min() >= 1
    assert result["nmfc_id"].max() <= 2
    peak_index = np.argmax(result["components"], axis=1)
    assert np.all(np.diff(peak_index) >= 0)


def test_fit_nmf_clusters_shapes_for_nonnegative_input():
    X_raw = np.array([
        [4.0, 1.0, 0.0, 0.0],
        [3.5, 0.8, 0.0, 0.0],
        [0.0, 0.0, 1.0, 4.0],
        [0.0, 0.0, 0.8, 3.5],
    ])
    X_nmf, component_n_bins = nmf.prepare_nonnegative_matrix(
        X_raw,
        neg_conversion="zero",
    )

    result = nmf.fit_nmf_clusters(
        X=X_nmf,
        sample_weights=np.ones(X_nmf.shape[0]),
        n_components=2,
        component_n_bins=component_n_bins,
        neg_conversion="zero",
        random_state=13,
    )

    assert result["components"].shape == (2, component_n_bins)
    assert result["signed_components"].shape == (2, component_n_bins)
    peak_index = np.argmax(result["components"], axis=1)
    assert np.all(np.diff(peak_index) >= 0)


def test_bootstrap_gap_nmf_summary_shapes():
    X = np.array([
        [4.0, 1.0, 0.0, 0.0],
        [3.5, 0.8, 0.0, 0.0],
        [3.8, 1.2, 0.0, 0.0],
        [0.0, 0.0, 1.0, 4.0],
        [0.0, 0.0, 0.8, 3.5],
        [0.0, 0.0, 1.2, 3.8],
    ])

    summary = nmf.bootstrap_gap_nmf(
        X=X,
        sample_weights=np.ones(X.shape[0]),
        k_list=(1, 2),
        n_boot=2,
        fraction=0.75,
        n_refs=2,
        n_jobs=1,
        random_state=13,
    )

    expected_keys = {
        "k_list",
        "full_gap",
        "full_gap_se",
        "full_dispersion",
        "full_log_ref_dispersion",
        "boot_gap",
        "boot_gap_se",
        "boot_optimal_k",
        "opti_k",
        "neg_conversion",
        "component_n_bins",
    }
    assert expected_keys == set(summary)
    assert summary["full_gap"].shape == (2,)
    assert summary["full_gap_se"].shape == (2,)
    assert summary["full_dispersion"].shape == (2,)
    assert summary["full_log_ref_dispersion"].shape == (2,)
    assert summary["boot_gap"].shape == (2, 2)
    assert summary["boot_gap_se"].shape == (2, 2)
    assert summary["boot_optimal_k"].shape == (2,)
    assert summary["opti_k"] in (1, 2)
    assert summary["neg_conversion"] == "zero"
    assert summary["component_n_bins"] == X.shape[1]


def test_bootstrap_gap_nmf_keeps_negative_values_by_splitting():
    X = np.array([
        [4.0, 1.0, -0.1, 0.0],
        [3.5, 0.8, -0.2, 0.0],
        [3.8, 1.2, -0.1, 0.0],
        [0.0, -0.2, 1.0, 4.0],
        [0.0, -0.1, 0.8, 3.5],
        [0.0, -0.2, 1.2, 3.8],
    ])

    summary = nmf.bootstrap_gap_nmf(
        X=X,
        k_list=(1, 2),
        n_boot=2,
        fraction=0.75,
        n_refs=2,
        n_jobs=1,
        random_state=13,
        neg_conversion="concat",
    )

    assert summary["neg_conversion"] == "concat"
    assert summary["component_n_bins"] == X.shape[1]
    assert summary["boot_gap"].shape == (2, 2)
    assert summary["opti_k"] in (1, 2)


def test_fit_nmf_clusters_can_prepare_and_search_k_internally_with_uniform_weights():
    X = np.array([
        [4.0, 1.0, -0.1, 0.0],
        [3.5, 0.8, -0.2, 0.0],
        [3.8, 1.2, -0.1, 0.0],
        [0.0, -0.2, 1.0, 4.0],
        [0.0, -0.1, 0.8, 3.5],
        [0.0, -0.2, 1.2, 3.8],
    ])

    result = nmf.fit_nmf_clusters(
        X=X,
        n_components=None,
        neg_conversion="concat",
        k_list=(1, 2),
        n_boot=2,
        fraction=0.75,
        n_refs=2,
        n_jobs=1,
        random_state=13,
    )

    assert result["neg_conversion"] == "concat"
    assert result["component_n_bins"] == X.shape[1]
    assert result["n_components"] in (1, 2)
    assert result["signed_components"].shape[1] == X.shape[1]
    assert result["nmfc_id"].shape == (X.shape[0],)
    assert result["gap_summary"]["opti_k"] == result["n_components"]


if __name__ == "__main__":
    test_prepare_nonnegative_matrix_zeroes_negative_input_by_default()
    test_prepare_nonnegative_matrix_splits_all_nonnegative_input_when_concat()
    test_prepare_nonnegative_matrix_zeroes_negative_input_explicitly()
    test_balanced_group_sample_weights()
    test_fit_nmf_clusters_shapes_and_component_sorting()
    test_fit_nmf_clusters_shapes_for_nonnegative_input()
    test_bootstrap_gap_nmf_summary_shapes()
    test_bootstrap_gap_nmf_keeps_negative_values_by_splitting()
    test_fit_nmf_clusters_can_prepare_and_search_k_internally_with_uniform_weights()
