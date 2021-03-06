from tempfile import NamedTemporaryFile

import pytest

from lhotse.cut import Cut, CutSet, MixedCut, MixTrack


@pytest.fixture
def cut_set_with_mixed_cut(cut1, cut2):
    mixed_cut = MixedCut(id='mixed-cut-id', tracks=[
        MixTrack(cut=cut1),
        MixTrack(cut=cut2, offset=1.0, snr=10)
    ])
    return CutSet({cut.id: cut for cut in [cut1, cut2, mixed_cut]})


def test_cut_set_iteration(cut_set_with_mixed_cut):
    cuts = list(cut_set_with_mixed_cut)
    assert len(cut_set_with_mixed_cut) == 3
    assert len(cuts) == 3


def test_cut_set_holds_both_simple_and_mixed_cuts(cut_set_with_mixed_cut):
    simple_cuts = cut_set_with_mixed_cut.simple_cuts.values()
    assert all(isinstance(c, Cut) for c in simple_cuts)
    assert len(simple_cuts) == 2
    mixed_cuts = cut_set_with_mixed_cut.mixed_cuts.values()
    assert all(isinstance(c, MixedCut) for c in mixed_cuts)
    assert len(mixed_cuts) == 1


def test_simple_cut_set_serialization(cut_set):
    with NamedTemporaryFile() as f:
        cut_set.to_yaml(f.name)
        restored = CutSet.from_yaml(f.name)
    assert cut_set == restored


def test_mixed_cut_set_serialization(cut_set_with_mixed_cut):
    with NamedTemporaryFile() as f:
        cut_set_with_mixed_cut.to_yaml(f.name)
        restored = CutSet.from_yaml(f.name)
    assert cut_set_with_mixed_cut == restored


def test_filter_cut_set(cut_set, cut1):
    filtered = cut_set.filter(lambda cut: cut.id == 'cut-1')
    assert len(filtered) == 1
    assert list(filtered)[0] == cut1
