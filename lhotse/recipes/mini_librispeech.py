import logging
import re
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Union

import torchaudio

from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

dataset_parts = ('dev-clean-2', 'train-clean-5')


def download_and_untar(
        target_dir: Pathlike = '.',
        force_download: Optional[bool] = False,
        url: Optional[str] = 'http://www.openslr.org/resources/31'
) -> None:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for part in dataset_parts:
        tar_name = f'{part}.tar.gz'
        tar_path = target_dir / tar_name
        if force_download or not tar_path.is_file():
            urllib.request.urlretrieve(f'{url}/{tar_name}', filename=tar_path)
        part_dir = target_dir / f'LibriSpeech/{part}'
        completed_detector = part_dir / '.completed'
        if not completed_detector.is_file():
            shutil.rmtree(part_dir, ignore_errors=True)
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=target_dir)
                completed_detector.touch()


class LibriSpeechMetaData(NamedTuple):
    audio_path: Pathlike
    audio_info: torchaudio.sox_signalinfo_t
    text: str


def prepare_mini_librispeech(
        corpus_dir: Pathlike,
        output_dir: Pathlike
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the yamls.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in dataset_parts:
        # Generate a mapping: utt_id -> (audio_path, audio_info, text)
        metadata = {}
        part_path = corpus_dir / part
        for trans_path in part_path.rglob('*.txt'):
            with open(trans_path) as f:
                for line in f:
                    idx, text = line.split(maxsplit=1)
                    audio_path = part_path / Path(idx.replace('-', '/')).parent / f'{idx}.flac'
                    if audio_path.is_file():
                        # info[0]: info of the raw audio (e.g. channel number, sample rate, duration ... )
                        # info[1]: info about the encoding (e.g. FLAC/ALAW/ULAW ...)
                        info = torchaudio.info(str(audio_path))
                        metadata[idx] = LibriSpeechMetaData(audio_path=audio_path, audio_info=info[0], text=text)
                    else:
                        logging.warning(f'No such file: {audio_path}')

        # Audio
        audio = RecordingSet.from_recordings(
            Recording(
                id=idx,
                sources=[
                    AudioSource(
                        type='file',
                        channel_ids=[0],
                        source=str(metadata[idx].audio_path)
                    )
                ],
                sampling_rate=int(metadata[idx].audio_info.rate),
                num_samples=metadata[idx].audio_info.length,
                duration_seconds=(metadata[idx].audio_info.length / metadata[idx].audio_info.rate)
            )
            for idx in metadata
        )
        audio.to_yaml(output_dir / f'audio_{part}.yml')

        # Supervision
        supervision = SupervisionSet.from_segments(
            SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=audio.recordings[idx].duration_seconds,
                channel_id=0,
                language='English',
                speaker=re.sub(r'-.*', r'', idx),
                text=metadata[idx].text.strip()
            )
            for idx in audio.recordings
        )
        supervision.to_yaml(output_dir / f'supervisions_{part}.yml')

        manifests[part] = {
            'audio': audio,
            'supervisions': supervision
        }

    return manifests
