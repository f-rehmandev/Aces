"""Unit tests for versioned datasets (spec §28)."""
from src.history.dataset import VersionedDataset, DatasetVersion


def test_new_dataset_is_empty():
    ds = VersionedDataset("q")
    assert len(ds) == 0
    assert ds.latest() is None
    assert ds.latest_valid() is None


def test_append_increments_version():
    ds = VersionedDataset("q")
    v1 = ds.append([{"title": "A"}])
    v2 = ds.append([{"title": "B"}])
    assert v1.version == 1
    assert v2.version == 2
    assert ds.latest().version == 2


def test_latest_valid_skips_failed_quality():
    ds = VersionedDataset("q")
    ds.append([{"title": "A"}], quality_passed=True)
    ds.append([{"title": "B"}], quality_passed=False)
    assert ds.latest_valid().version == 1


def test_latest_valid_skips_superseded():
    ds = VersionedDataset("q")
    ds.append([{"title": "A"}], quality_passed=True)
    ds.append([{"title": "B"}], quality_passed=True)
    ds.supersede(1)
    assert ds.latest_valid().version == 2


def test_get_returns_specific_version():
    ds = VersionedDataset("q")
    ds.append([{"title": "A"}])
    ds.append([{"title": "B"}])
    assert ds.get(1).records == [{"title": "A"}]
    assert ds.get(2).records == [{"title": "B"}]
    assert ds.get(99) is None


def test_supersede_unknown_raises():
    import pytest
    ds = VersionedDataset("q")
    with pytest.raises(KeyError):
        ds.supersede(5)


def test_to_dict_and_from_dict_roundtrip():
    ds = VersionedDataset("my-task")
    ds.append([{"title": "A"}], quality_score=0.9, confidence_mean=0.8)
    ds.append([{"title": "B"}], quality_passed=False)

    d = ds.to_dict()
    ds2 = VersionedDataset.from_dict(d)
    assert ds2.task_key == "my-task"
    assert len(ds2) == 2
    assert ds2.get(1).quality_score == 0.9
    assert ds2.get(2).quality_passed is False


def test_all_versions_returns_copy():
    ds = VersionedDataset("q")
    ds.append([{"title": "A"}])
    versions = ds.all_versions()
    versions.append("garbage")
    assert len(ds) == 1