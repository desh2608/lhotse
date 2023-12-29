from collections import defaultdict

import numpy as np
import pytest
from torch.utils.data import DataLoader

from lhotse.cut import CutSet
from lhotse.dataset.sampling import SimpleCutSampler
from lhotse.dataset.surt import K2SurtDataset
from lhotse.utils import compute_num_frames


@pytest.fixture
def cut_set():
    return CutSet.from_shar(in_dir="test/fixtures/lsmix")


@pytest.mark.parametrize("num_workers", [0, 1])
@pytest.mark.parametrize("return_sources", [True, False])
@pytest.mark.parametrize("max_prefix_speakers", [0, 4])
def test_surt_iterable_dataset(
    cut_set, num_workers, return_sources, max_prefix_speakers
):
    dataset = K2SurtDataset(
        return_sources=return_sources,
        return_cuts=True,
        max_prefix_speakers=max_prefix_speakers,
        speaker_buffer_frames=[10],
    )
    sampler = SimpleCutSampler(cut_set, shuffle=False, max_cuts=10000)
    # Note: "batch_size=None" disables the automatic batching mechanism,
    #       which is required when Dataset takes care of the collation itself.
    dloader = DataLoader(
        dataset, batch_size=None, sampler=sampler, num_workers=num_workers
    )
    batch = next(iter(dloader))
    assert batch["inputs"].shape == (2, 2238, 80)
    assert batch["input_lens"].tolist() == [2238, 985]

    assert len(batch["supervisions"][1]) == 2
    assert len(batch["text"][1]) == 2
    assert batch["text"][1] == [
        "BY THIS MANOEUVRE WE DON'T LET ANYBODY IN THE CAR AND WE TRY AND KEEP THEM CLEAR OF THE CAR SHORT OF SHOOTING THEM THAT IS CARRIED NO OTHER MESSAGE",
        "THE AMERICAN INTERPOSED BRUSQUELY BETWEEN PAROXYSMS AND THEY CAUGHT HIM AT IT EH",
    ]
    if return_sources:
        assert len(batch["source_feats"]) == 2
        assert all(
            len(batch["source_feats"][i]) == len(batch["cuts"][i].supervisions)
            for i in range(2)
        )
    if max_prefix_speakers > 0:
        assert batch["speaker_prefix"] is not None
        assert len(batch["speaker_prefix"]) == 2
        prefix = batch["speaker_prefix"][1]
        num_prefix_speakers = batch["num_prefix_speakers"]
        buffers = []
        for i in range(num_prefix_speakers):
            buffers.append(prefix[i * 10 : (i + 1) * 10])
        cut = batch["cuts"][1]
        feats = cut.load_features()
        speaker_feats = defaultdict(list)
        for channel in range(2):
            for sup, spk in zip(
                batch["supervisions"][1][channel], batch["speakers"][1][channel]
            ):
                start_frame = compute_num_frames(
                    sup.start, cut.frame_shift, cut.sampling_rate
                )
                end_frame = compute_num_frames(
                    sup.end, cut.frame_shift, cut.sampling_rate
                )
                feat = feats[start_frame:end_frame]
                speaker_feats[spk].append(feat)
        speaker_feats = {
            spk: np.concatenate(feats, axis=0) for spk, feats in speaker_feats.items()
        }
        assert all(
            i + 1 not in speaker_feats or buffer.numpy() in speaker_feats[i + 1]
            for i, buffer in enumerate(buffers)
        )
