"""
This is a data preparation script for the Mixer 6 dataset. The following description
is taken from the LDC website:

Mixer 6 Speech comprises 15,863 hours of audio recordings of interviews, transcript 
readings and conversational telephone speech involving 594 distinct native English 
speakers. This material was collected by LDC in 2009 and 2010 as part of the Mixer 
project, specifically phase 6, the focus of which was on native American English 
speakers local to the Philadelphia area.

The telephone collection protocol was similar to other LDC telephone studies (e.g., 
Switchboard-2 Phase III Audio - LDC2002S06): recruited speakers were connected through 
a robot operator to carry on casual conversations lasting up to 10 minutes, usually 
about a daily topic announced by the robot operator at the start of the call. The raw 
digital audio content for each call side was captured as a separate channel, and each 
full conversation was presented as a 2-channel interleaved audio file, with 8000 
samples/second and u-law sample encoding. Each speaker was asked to complete 15 calls.

The multi-microphone portion of the collection utilized 14 distinct microphones 
installed identically in two mutli-channel audio recording rooms at LDC. Each session 
was guided by collection staff using prompting and recording software to conduct the 
following activities: (1) repeat questions (less than one minute), (2) informal 
conversation (typically 15 minutes), (3) transcript reading (approximately 15 minutes) 
and (4) telephone call (generally 10 minutes). Speakers recorded up to three 45-minute 
sessions on distinct days. The 14 channels were recorded synchronously into separate 
single-channel files, using 16-bit PCM sample encoding at 16000 samples/second.

The collection contains 4,410 recordings made via the public telephone network and 
1,425 sessions of multiple microphone recordings in office-room settings. The telephone 
recordings are presented as 8-KHz 2-channel NIST SPHERE files, and the microphone 
recordings are 16-KHz 1-channel flac/ms-wav files. 
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union, NamedTuple, Tuple

from tqdm import tqdm

from lhotse import validate_recordings_and_supervisions, fix_manifests
from lhotse.audio import Recording, RecordingSet, AudioSource, sph_info
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, is_module_available


ERROR_WORDS = [
    "DELETD",
    "DELETE",
    "DELETED",
    "DELETES",
    "DELETEd",
    "DELETeD",
    "DELTED",
    "ELETED",
    "hDELETED",
]


class MixerSegmentAnnotation(NamedTuple):
    session: str
    speaker: str
    start: Seconds
    end: Seconds
    text: str


def _read_mx6_subj_info(corpus_dir: Pathlike) -> Dict[str, Dict[str, str]]:
    """
    The docs contain the following fields:

     1  subjid - numeric identifier, links to calls and interviews
     2  sex - M or F
     3  yob - year of birth
     4  edu_years - years of formal education
     5  edu_degree - highest education degree earned
     6  edu_deg_yr - year in which highest degree was earned
     7  edu_contig - Y or N: were all edu_years spent contiguously?
     8  esl_age - for ESL speakers, age when English was learned
     9  ntv_lg - native language (ISO 639-3 code)
    10  oth_lgs - other languages (ISO 639-3 codes, '/'-separated)
    11  occup - occupation
    12  cntry_born - country where born
    13  state_born - state where born
    14  city_born - city where born
    15  cntry_rsd - country where raised
    16  state_rsd - state where raised
    17  city_rsd - city where raised
    18  ethnic - ethnicity
    19  smoker - Y or N
    20  ht_cm - height in centimeters
    21  wt_kg - weight in kilograms
    22  mother_born - country (state city) where mother was born
    23  mother_raised - country (state city) where mother was raised
    24  mother_lang - mother's native language
    25  mother_edu - mother's years of formal education
    26  father_born - country (state city) where father was born
    27  father_raised - country (state city) where father was raised
    28  father_lang - father's native language
    29  father_edu - father's years of formal education
    
    But here we only store the `sex`. If other fields are needed, they can be read and stored
    in the `custom` field of the supervision set.
    """
    subj_info = {}
    with open(corpus_dir / "docs" / "mx6_subjs.csv", "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            subj_info[parts[0]] = {"sex": parts[1]}
    return subj_info

def _read_mx6_ivcomponents(corpus_dir: Pathlike) -> Dict[str, Dict[str, Tuple[float]]]:
    """
    Read components. Each session is divided into 4 parts: repeating questions (rptq), 
    interview (intv), prompt reading (rdtr), and call (call). The LDC catalog provides 
    the time markers for start and end of each part.
    """
    iv_components = {}
    with open(corpus_dir / "docs" / "mx6_ivcomponents.csv", "r") as f:
        next(f)
        for l in f:
            session_id, rptq_bgn, rptq_end, intv_bgn, intv_end, rdtr_bgn, rdtr_end, call_bgn, call_end, call_type = l.strip().split(",")
            iv_components[session_id] = {
                "rptq": (float(rptq_bgn), float(rptq_end)),
                "intv": (float(intv_bgn), float(intv_end)),
                "rdtr": (float(rdtr_bgn), float(rdtr_end)),
                "call": (float(call_bgn), float(call_end)),
            }
    return iv_components

def _read_mx6_transcript_sentences(corpus_dir: Pathlike) -> str:
    """
    Read the transcript sentences that were used for the prompt reading part of the session.
    We return these as a single string since the corpus does not provide alignments.
    NOTE: If the speaker got to the end of the list quickly, with time remaining in the session
    schedule for transcript reading, the list was simply presented again, starting over 
    at the first sentence.  (So, some sessions contain more than 335 sentence readings 
    in this component, and in this case, sentences at the start of the list will have been 
    read twice.)
    """
    transcript = []
    with open(corpus_dir / "docs" / "mx6_transcript_sentences.txt", "r") as f:
        next(f)
        for l in f:
            # Remove the number at the beginning of each line.
            transcript.append(l.strip().split(".", 1)[1])
    return " ".join(transcript)

def _read_mx6_calls(corpus_dir: Pathlike) -> Dict[str, Tuple[str]]:
    """
    Read the metadata for the call part of the session. The LDC catalog provides the following fields:

     1  call_id - numeric identifier, links to audio file name
     2  call_date - links to audio file name
     3  lang - language in which the conversation was conducted
     4  eng_stat - one of: AllENG, SomeENG, NoENG
     5  sid_a - subjid of the speaker  channel A
     6  phid_a - telephone ID on channel A
     7  ph_categ_a - one of: M (main phone), O (other phone)
     8  phtyp_a - one of: 1 (cell phone), 2 (cordless), 3 (standard)
     9  phmic_a - one of: 1 (spkr-phone), 2 (headset), 3 (earbud), 4 (handheld)
    10  cnvq_a - audit judgment of conversation quality (Good,Acceptable,Unsuitable)
    11  sigq_a - audit judgment of signal quality (Good,Acceptable,Unsuitable)
    12  tbug_a - Y or N: auditor found a technical problem channel A
    13-20 - same as 5-12, applied to channel B
    21  topic - numeric ID of the topic announced to the callers
        (refer to mx6_collection_doc.pdf for the numbered list of topics)

    Here, we only store `sid_a` and `sid_b` (the subjid of the speaker on channel A and B).
    """
    calls = defaultdict(tuple)
    with open(corpus_dir / "docs" / "mx6_calls.csv", "r") as f:
        next(f)
        for l in f:
            parts = l.strip().split(",")
            calls[parts[0]] = (parts[4], parts[12])
    return calls

def _read_mx6_intvs(corpus_dir) -> Dict[str, Tuple[Union[str, float]]]:
    """
    Read the metadata for the sessions. The LDC catalog provides the following fields:

     1  subj_id - numeric identifier, links to subjects table
     2  session_fileid - audio file name, includes date, time, location
     3  duration - in seconds, for the entire session recording
     4  interviewer_id - subjid of LDC staff person conducting the session
     5  call_type - one of: high_ve, low_ve, cell, normal
     6  call_id - numeric identifier, links to calls table
     7  call_chan - A or B: side of 2-channel ulaw audio matching IV audio
     8  wb_tconv_offset - seconds from start of IV audio where call begins
    
    Here, we only store (subj_id, intv_id, duration) as a tuple.
    """
    intvs = {}
    with open(corpus_dir / "docs" / "mx6_intvs.csv", "r") as f:
        next(f)
        for l in f:
            parts = l.strip().split(",")
            intvs[parts[1]] = (parts[0], parts[3], float(parts[2]))
    return intvs

def prepare_mixer6(
    corpus_dir: Pathlike,
    transcript_dir: Optional[Pathlike],
    output_dir: Optional[Pathlike] = None,
    part: str = "intv",
    channels: list = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    mixed_cuts: bool = True,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the speech corpus dir (LDC2013S03).
    :param transcript_dir: Pathlike, the path of the transcript dir (Mixer6Transcription).
    :param output_dir: Pathlike, the path where to write the manifests.
    :param part: str, "call_sph", "call_flac", "intv", "rptq", or "rdtr", specifies whether to prepare the
        call (sphere files), call (room audio), or interview data.
    :param channels: list, the list of channel integer ids to include as sources.
        By default we exclude channels CH01 (0), CH03 (2), and CH13 (13).
    :return: a Dict whose key is the dataset part ('dev' and 'dev_test'), and the value
        is Dicts with the keys 'recordings' and 'supervisions'.

    NOTE on interview data: each recording in the interview data contains 14 channels. Channel 0 (Mic 01)
    is the lapel mic for the interviewer, so it can be treated as close-talk. Channel 1 (Mic 02) is
    the lapel mic for the interviewee.  All other mics are placed throughout the room.
    Channels 2 and 13 (Mics 03 and 14) are often silent, and so they may be removed.

    NOTE: the official LDC corpus does not contain transcriptions for the data.
    """
    if not is_module_available("textgrid"):
        raise ValueError(
            "To prepare Mixer 6 data, please 'pip install textgrid' first."
        )
    import textgrid

    corpus_dir = Path(corpus_dir)
    transcript_dir = Path(transcript_dir)
    
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    assert part in ["call_sph", "call_flac", "intv", "rptq", "rdtr"], (
        f"Invalid part: {part}. "
        "Valid values are 'call_sph', 'call_flac', 'intv', 'rptq', 'rdtr'."
    )

    # Get subject info
    subj_info = _read_mx6_subj_info(corpus_dir)
    # Get session metadata
    iv_components = _read_mx6_ivcomponents(corpus_dir)
    iv_metadata = _read_mx6_intvs(corpus_dir)

    # Prepare recordings
    recordings = []
    if part == "call_sph":
        # SPH files have names such as 20100130_170728_4005.sph
        for path in tqdm(
            list(corpus_dir.rglob("*.sph")), desc="Processing call sphere files"
        ):
            recordings.append(Recording.from_file(path))
    else:
        # FLAC files have names such as 20100113_092658_LDC_120840_CH01.flac
        
        # First we go over all the FLAC files and group them by session.
        reco_to_channels = defaultdict(dict)
        for path in tqdm(corpus_dir.rglob("*.flac"), desc="Processing flac files"):
            reco_id = path.stem
            channel = int(reco_id.split("_")[-1][2:])
            if channel in channels:
                reco_to_channels[reco_id].update({channel: path})
        
        # Now create a Recording for each session.
        import soundfile as sf
        for reco_id, paths in tqdm(reco_to_channels.items(), desc="Processing sessions"):
            if len(paths) != len(channels):
                logging.warning(
                    f"Expected {len(channels)} channels, but found {len(paths)} channels for reco {reco_id}. Skipping."
                )
                continue
            audio_sf = sf.SoundFile(paths[channels[0]])
            recordings.append(
                Recording(
                    id=reco_id,
                    sources=[AudioSource(type="file", channels=[c], source=p) for c, p in paths.items()],
                    sampling_rate=int(audio_sf.samplerate),
                    num_samples=int(audio_sf.frames),
                    duration=float(audio_sf.frames) / audio_sf.samplerate,
                )
            )

    # Prepare supervisions
    supervisions = []
    if part == "call_sph":
        pass
    elif part == "call_flac":
        pass
    elif part == "intv":
        pass
    elif part == "rptq" or part == "rdtr":
        text = _read_mx6_transcript_sentences(corpus_dir) if part == "rdtr" else ""
        for session_id, components in iv_components:
            beg, end = components[part]
            speaker_id = session_id.split("_")[-1]
            supervisions.extend(
                [SupervisionSegment(
                    id=session_id,
                    recording_id=session_id,
                    start=beg,
                    duration=end - beg,
                    channel=c,
                    language="English",
                    speaker=speaker_id,
                    text=text,
                ) for c in channels]
            )
        intv_list = {}
        with open(corpus_dir / "docs" / "mx6_intvs.csv", "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                spk_id = parts[0]
                session_time = parts[1].split("_")[0]
                intv_list[f"{spk_id}_{session_time}"] = parts[1]

        text_paths = list(transcript_dir.rglob(f"intv/*rid"))
        for text_path in tqdm(text_paths, desc="Processing intv textgrids"):
            try:
                spk_id, session_time, _, _ = text_path.stem.split("_")
            except ValueError:
                logging.warning(f"Skipping {text_path.stem}")
                continue
            intv = f"{spk_id}_{session_time}"
            audio_id = intv_list[intv]
            tg = textgrid.TextGrid.fromFile(str(text_path))
            for i, tier in enumerate(tg.tiers):
                for j, interval in enumerate(tier.intervals):
                    if interval.mark != "" and not any(
                        w in ERROR_WORDS for w in interval.mark.split()
                    ):
                        start = interval.minTime
                        end = interval.maxTime
                        text = " ".join(interval.mark.split(" ")[1:])
                        channel_iterator = channels if not mixed_cuts else [channels[0]]
                        for chn in channel_iterator:
                            filename = f"{audio_id}_CH{chn+1:02d}.flac"
                            file_source = (
                                corpus_dir
                                / "data"
                                / "pcm_flac"
                                / f"CH{chn+1:02d}"
                                / filename
                            )
                            if file_source.is_file():
                                segment = SupervisionSegment(
                                    id=f"{intv}-{i}-{j}-{chn}",
                                    recording_id=audio_id,
                                    start=start + reco_to_offsets[audio_id][0][0],
                                    duration=round(end - start, 4),
                                    channel=chn,
                                    language="English",
                                    speaker=f"{spk_id}-{i}",
                                    text=text,
                                )
                                supervisions.append(segment)

    # Prepare supervisions
    supervisions = []
    if "call " in part:
        text_paths = transcript_dir.rglob("*/call/*.textgrid")
        for text_path in tqdm(text_paths, desc="Processing call textgrids"):
            session_id = "_".join(text_path.stem.split("_")[:-2])
            speaker_id = session_id.split("_")[-1]
            tg = textgrid.TextGrid.fromFile(str(text_path))
            for i in range(len(tg.tiers)):
                for j in range(len(tg.tiers[i].intervals)):
                    if tg.tiers[i].intervals[j].mark != "" and not any(
                        w in ERROR_WORDS for w in tg.tiers[i].intervals[j].mark.split()
                    ):
                        start = tg.tiers[i].intervals[j].minTime
                        end = tg.tiers[i].intervals[j].maxTime
                        text = " ".join(tg.tiers[i].intervals[j].mark.split(" ")[1:])
                        segment = SupervisionSegment(
                            id=f"{session_id}-{i}-{j}",
                            recording_id=session_id,
                            start=start + reco_to_offsets[session_id][1][0],
                            duration=round(end - start, 4),
                            channel=0,
                            language="English",
                            speaker=f"{speaker_id}-{i}",
                            text=text,
                        )
                        supervisions.append(segment)
    else:
        





    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    if output_dir is not None:
        recording_set.to_file(output_dir / f"recordings.jsonl")
        supervision_set.to_file(output_dir / f"supervisions.jsonl")

    manifests = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
