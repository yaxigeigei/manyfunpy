import gzip
import pickle

from manyfunpy.io import load_pickle, save_pickle


def test_save_and_load_pickle_round_trip(tmp_path):
    # Save and reload an uncompressed pickle.
    path = tmp_path / "payload.pkl"
    payload = {"labels": ["a", "b"], "values": [1, 2, 3]}

    save_pickle(payload, path)

    assert load_pickle(path) == payload


def test_save_and_load_gzip_pickle_by_extension(tmp_path):
    # Save and reload a gzip-compressed pickle.
    path = tmp_path / "payload.pkl.gz"
    payload = {"labels": ["a", "b"], "values": [1, 2, 3]}

    save_pickle(payload, path)

    with gzip.open(path, "rb") as stream:
        assert pickle.load(stream) == payload
    assert load_pickle(path) == payload
