# """
# Note: this module is very heavily based on a torchaudio tutorial about forced
# alignment with the multilingual MMS model.

# Link: https://pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html
# """
# import logging
# import os
# import re
# import tempfile
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Callable, Dict, Generator, List, Tuple

# import torch
# import torchaudio
# from torchaudio.models import wav2vec2_model

# from lhotse import CutSet, MonoCut
# from lhotse.supervision import AlignmentItem
# from lhotse.utils import Pathlike


# TORCHAUDIO_MMS_MODEL_CKPT = (
#     "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt"
# )


# def align_with_mms(
#     cuts: CutSet,
#     device: str = "cpu",
#     uroman_root_dir: Pathlike = None,
# ) -> Generator[MonoCut, None, None]:
#     """
#     Use the multilingual MMS model (https://arxiv.org/abs/2305.13516) to perform forced
#     word-level alignment of a CutSet.

#     This means that for every SupervisionSegment with a transcript, we will find the
#     start and end timestamps for each of the words.

#     We support cuts with multiple supervisions -- the forced alignment will be done
#     for every supervision region separately, and attached to the relevant supervision.

#     .. hint::

#         If your segments are not accurate, you may want to merge all segments together
#         before running force-alignment. See :func:`lhotse.cut.set.merge_supervisions`.

#     .. warning::

#         This function internally runs a perl script from the uroman toolkit.

#     :param cuts: input CutSet.
#     :param device: device on which to run the computation. Defaults to "cpu".
#     :return: a generator of cuts that have the "alignment" field set in each of
#         their supervisions.
#     """
#     try:
#         from torchaudio.functional import forced_align
#     except ModuleNotFoundError:
#         print(
#             "Failed to import the forced alignment API. "
#             "Please install torchaudio nightly builds. "
#             "Please refer to https://pytorch.org/get-started/locally "
#             "for instructions to install a nightly build."
#         )
#         raise

#     assert uroman_root_dir is not None, (
#         "Please provide the path to the uroman root directory. "
#         "You can download it from https://github.com/isi-nlp/uroman. "
#     )

#     model = _get_model(device=device)
#     dictionary = _get_dictionary()

#     for cut in cuts:
#         for idx, subcut in enumerate(cut.trim_to_supervisions(keep_overlapping=False)):
#             sup = subcut.supervisions[0]
#             if sup.text is None or len(sup.text) == 0:
#                 continue

#             waveform = torch.as_tensor(subcut.load_audio(), device=device)
#             # Ratio of number of samples to number of frames
#             ratio = waveform.size(1) / emission.size(0)
#             alignment = [
#                 AlignmentItem(
#                     symbol=ws.label,
#                     start=round(
#                         subcut.start + int(ratio * ws.start) / sampling_rate, ndigits=8
#                     ),
#                     duration=round(
#                         int(subcut.start + ratio * (ws.end - ws.start)) / sampling_rate,
#                         ndigits=8,
#                     ),
#                     score=ws.score,
#                 )
#                 for ws in word_segments
#             ]

#             # Important: reference the original supervision before "trim_to_supervisions"
#             #            because the new one has start=0 to match the start of the subcut
#             sup = cut.supervisions[idx].with_alignment(kind="word", alignment=alignment)
#             cut.supervisions[idx] = sup

#         yield cut


# def _get_model(device: str = "cpu") -> torch.nn.Module:
#     model = wav2vec2_model(
#         extractor_mode="layer_norm",
#         extractor_conv_layer_config=[
#             (512, 10, 5),
#             (512, 3, 2),
#             (512, 3, 2),
#             (512, 3, 2),
#             (512, 3, 2),
#             (512, 2, 2),
#             (512, 2, 2),
#         ],
#         extractor_conv_bias=True,
#         encoder_embed_dim=1024,
#         encoder_projection_dropout=0.0,
#         encoder_pos_conv_kernel=128,
#         encoder_pos_conv_groups=16,
#         encoder_num_layers=24,
#         encoder_num_heads=16,
#         encoder_attention_dropout=0.0,
#         encoder_ff_interm_features=4096,
#         encoder_ff_interm_dropout=0.1,
#         encoder_dropout=0.0,
#         encoder_layer_norm_first=True,
#         encoder_layer_drop=0.1,
#         aux_num_out=31,
#     )

#     model.load_state_dict(
#         torch.hub.load_state_dict_from_url(
#             TORCHAUDIO_MMS_MODEL_CKPT, map_location=device
#         )
#     )
#     model.eval()
#     return model


# def _get_emission(model: torch.nn.Module, waveform: torch.Tensor) -> torch.Tensor:
#     # NOTE: this step is essential
#     waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)

#     emissions, _ = model(waveform)
#     emissions = torch.log_softmax(emissions, dim=-1)
#     emission = emissions[0].cpu().detach()

#     # Append the extra dimension corresponding to the <star> token
#     extra_dim = torch.zeros(emissions.shape[0], emissions.shape[1], 1)
#     emissions = torch.cat((emissions.cpu(), extra_dim), 2)
#     emission = emissions[0].detach()
#     return emission, waveform


# def _get_dictionary() -> Dict:
#     # Construct the dictionary
#     # '@' represents the OOV token, '*' represents the <star> token.
#     # <pad> and </s> are fairseq's legacy tokens, which're not used.
#     dictionary = {
#         "<blank>": 0,
#         "<pad>": 1,
#         "</s>": 2,
#         "@": 3,
#         "a": 4,
#         "i": 5,
#         "e": 6,
#         "n": 7,
#         "o": 8,
#         "u": 9,
#         "t": 10,
#         "s": 11,
#         "r": 12,
#         "m": 13,
#         "k": 14,
#         "l": 15,
#         "d": 16,
#         "g": 17,
#         "h": 18,
#         "y": 19,
#         "b": 20,
#         "p": 21,
#         "w": 22,
#         "c": 23,
#         "v": 24,
#         "j": 25,
#         "z": 26,
#         "f": 27,
#         "'": 28,
#         "q": 29,
#         "x": 30,
#         "*": 31,
#     }
#     return dictionary


# @dataclass
# class Frame:
#     # This is the index of each token in the transcript,
#     # i.e. the current frame aligns to the N-th character from the transcript.
#     token_index: int
#     time_index: int
#     score: float


# @dataclass
# class Segment:
#     label: str
#     start: int
#     end: int
#     score: float

#     def __repr__(self):
#         return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

#     @property
#     def length(self):
#         return self.end - self.start


# # compute frame-level and word-level alignments using torchaudio's forced alignment API
# def _compute_alignments(
#     transcript: str, dictionary: Dict, emission: torch.Tensor, forced_align_fn: Callable
# ) -> Tuple[List[Segment], List[str], int]:
#     frames = []
#     tokens = [dictionary[c] for c in transcript.replace(" ", "")]

#     targets = torch.tensor(tokens, dtype=torch.int32)
#     input_lengths = torch.tensor(emission.shape[0])
#     target_lengths = torch.tensor(targets.shape[0])

#     # This is the key step, where we call the forced alignment API functional.forced_align to compute frame alignments.
#     frame_alignment, frame_scores = forced_align_fn(
#         emission, targets, input_lengths, target_lengths, 0
#     )

#     assert len(frame_alignment) == input_lengths.item()
#     assert len(targets) == target_lengths.item()

#     token_index = -1
#     prev_hyp = 0
#     for i in range(len(frame_alignment)):
#         if frame_alignment[i].item() == 0:
#             prev_hyp = 0
#             continue

#         if frame_alignment[i].item() != prev_hyp:
#             token_index += 1
#         frames.append(Frame(token_index, i, frame_scores[i].exp().item()))
#         prev_hyp = frame_alignment[i].item()

#     # compute frame alignments from token alignments
#     transcript_nospace = transcript.replace(" ", "")
#     i1, i2 = 0, 0
#     segments = []
#     while i1 < len(frames):
#         while i2 < len(frames) and frames[i1].token_index == frames[i2].token_index:
#             i2 += 1
#         score = sum(frames[k].score for k in range(i1, i2)) / (i2 - i1)

#         segments.append(
#             Segment(
#                 transcript_nospace[frames[i1].token_index],
#                 frames[i1].time_index,
#                 frames[i2 - 1].time_index + 1,
#                 score,
#             )
#         )
#         i1 = i2

#     # compue word alignments from token alignments
#     separator = " "
#     words = []
#     i1, i2, i3 = 0, 0, 0
#     while i3 < len(transcript):
#         if i3 == len(transcript) - 1 or transcript[i3] == separator:
#             if i1 != i2:
#                 if i3 == len(transcript) - 1:
#                     i2 += 1
#                 s = 0
#                 segs = segments[i1 + s : i2 + s]
#                 word = "".join([seg.label for seg in segs])
#                 score = sum(seg.score * seg.length for seg in segs) / sum(
#                     seg.length for seg in segs
#                 )
#                 words.append(
#                     Segment(
#                         word, segments[i1 + s].start, segments[i2 + s - 1].end, score
#                     )
#                 )
#             i1 = i2
#         else:
#             i2 += 1
#         i3 += 1

#     num_frames = len(frame_alignment)
#     return segments, words, num_frames


# # The following are taken from:
# # https://github.com/facebookresearch/fairseq/blob/main/examples/mms/data_prep/align_utils.py

# # iso codes with specialized rules in uroman
# # fmt: off
# special_isos_uroman = [
#     "ara", "bel", "bul", "deu", "ell", "eng", "fas", "grc", "ell", "eng", "heb", "kaz",
#     "kir", "lav", "lit", "mkd", "mkd2", "oss", "pnt", "pus", "rus", "srp", "srp2", "tur",
#     "uig", "ukr", "yid",
# ]
# # fmt: on


# def _normalize_uroman(text: str):
#     text = text.lower()
#     text = re.sub("([^a-z' ])", " ", text)
#     text = re.sub(" +", " ", text)
#     return text.strip()


# def _get_uroman_tokens(
#     norm_transcripts: List[str], uroman_root_dir: Pathlike, iso=None
# ):
#     tf = tempfile.NamedTemporaryFile()
#     tf2 = tempfile.NamedTemporaryFile()
#     with open(tf.name, "w") as f:
#         for t in norm_transcripts:
#             f.write(t + "\n")

#     assert Path(f"{uroman_root_dir}/uroman.pl").exists(), "uroman not found"
#     cmd = f"perl {uroman_root_dir}/uroman.pl"
#     if iso in special_isos_uroman:
#         cmd += f" -l {iso} "
#     cmd += f" < {tf.name} > {tf2.name}"
#     os.system(cmd)
#     outtexts = []
#     with open(tf2.name) as f:
#         for line in f:
#             line = " ".join(line.strip())
#             line = re.sub(r"\s+", " ", line).strip()
#             outtexts.append(line)
#     assert len(outtexts) == len(norm_transcripts)
#     uromans = []
#     for ot in outtexts:
#         uromans.append(_normalize_uroman(ot))
#     return uromans
