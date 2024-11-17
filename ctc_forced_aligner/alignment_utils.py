import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
from packaging import version
from transformers import AutoModelForCTC, AutoTokenizer
from transformers import __version__ as transformers_version
from transformers.utils import is_flash_attn_2_available
from ._ctc_forced_align import forced_align as forced_align_cpp
from typing import List

SAMPLING_FREQ = 16000


@dataclass
class Segment:
    label: str
    start: int
    end: int

    def __repr__(self):
        return f"{self.label}: [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, idx_to_token_map):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1] == path[i2]:
            i2 += 1
        segments.append(Segment(idx_to_token_map[path[i1]], i1, i2 - 1))
        i1 = i2
    return segments


def time_to_frame(time):
    stride_msec = 20
    frames_per_sec = 1000 / stride_msec
    return int(time * frames_per_sec)


def get_spans(tokens, segments, blank):
    ltr_idx = 0
    tokens_idx = 0
    intervals = []
    start, end = (0, 0)
    for seg_idx, seg in enumerate(segments):
        if tokens_idx == len(tokens):
            assert seg_idx == len(segments) - 1
            assert seg.label == blank
            continue
        cur_token = tokens[tokens_idx].split(" ")
        ltr = cur_token[ltr_idx]
        if seg.label == blank:
            continue
        assert seg.label == ltr, f"{seg.label} != {ltr}"
        if (ltr_idx) == 0:
            start = seg_idx
        if ltr_idx == len(cur_token) - 1:
            ltr_idx = 0
            tokens_idx += 1
            intervals.append((start, seg_idx))
            while tokens_idx < len(tokens) and len(tokens[tokens_idx]) == 0:
                intervals.append((seg_idx, seg_idx))
                tokens_idx += 1
        else:
            ltr_idx += 1
    spans = []
    for idx, (start, end) in enumerate(intervals):
        span = segments[start : end + 1]
        if start > 0:
            prev_seg = segments[start - 1]
            if prev_seg.label == blank:
                pad_start = (
                    prev_seg.start
                    if (idx == 0)
                    else int((prev_seg.start + prev_seg.end) / 2)
                )
                span = [Segment(blank, pad_start, span[0].start)] + span
        if end + 1 < len(segments):
            next_seg = segments[end + 1]
            if next_seg.label == blank:
                pad_end = (
                    next_seg.end
                    if (idx == len(intervals) - 1)
                    else math.floor((next_seg.start + next_seg.end) / 2)
                )
                span = span + [Segment(blank, span[-1].end, pad_end)]
        spans.append(span)
    return spans


def load_audio(audio_file: str, dtype: torch.dtype, device: str):
    waveform, audio_sf = torchaudio.load(audio_file)  # waveform: channels X T
    waveform = torch.mean(waveform, dim=0)

    if audio_sf != SAMPLING_FREQ:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=audio_sf, new_freq=SAMPLING_FREQ
        )
    waveform = waveform.to(dtype).to(device)
    return waveform


def generate_emissions(
    model,
    audio_waveform: torch.Tensor,
    window_length=30,
    context_length=2,
    batch_size=4,
):
    # batching the input tensor and including a context before and after the input tensor

    batch_size = min(batch_size, 1)
    context = context_length * SAMPLING_FREQ
    window = window_length * SAMPLING_FREQ
    extention = math.ceil(
        audio_waveform.size(0) / window
    ) * window - audio_waveform.size(0)
    padded_waveform = torch.nn.functional.pad(
        audio_waveform, (context, context + extention)
    )
    input_tensor = padded_waveform.unfold(0, window + 2 * context, window)

    # Batched Inference
    emissions_arr = []
    with torch.inference_mode():
        for i in range(0, input_tensor.size(0), batch_size):
            input_batch = input_tensor[i : i + batch_size]
            emissions_ = model(input_batch).logits
            emissions_arr.append(emissions_)

    emissions = torch.cat(emissions_arr, dim=0)[
        :,
        time_to_frame(context_length) : -time_to_frame(context_length) + 1,
    ]  # removing the context
    emissions = emissions.flatten(0, 1)
    if time_to_frame(extention / SAMPLING_FREQ) > 0:
        emissions = emissions[: -time_to_frame(extention / SAMPLING_FREQ), :]

    emissions = torch.log_softmax(emissions, dim=-1)
    emissions = torch.cat(
        [emissions, torch.zeros(emissions.size(0), 1).to(emissions.device)], dim=1
    )  # adding a star token dimension
    stride = float(audio_waveform.size(0) * 1000 / emissions.size(0) / SAMPLING_FREQ)

    return emissions, math.ceil(stride)

def forced_align(
    log_probs: np.ndarray,
    targets: np.ndarray,
    input_lengths: Optional[np.ndarray] = None,
    target_lengths: Optional[np.ndarray] = None,
    blank: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if blank in targets:
        raise ValueError(
            f"Targets shouldn't contain the blank index. Found {targets}."
        )
    if blank >= log_probs.shape[-1] or blank < 0:
        raise ValueError("Blank index must be within [0, log_probs.shape[-1])")
    if np.max(targets) >= log_probs.shape[-1] or np.min(targets) < 0:
        raise ValueError("Target values must be within [0, log_probs.shape[-1])")
    assert log_probs.dtype == np.float32, "log_probs must be float32"

    if input_lengths is None:
        input_lengths = np.full(log_probs.shape[0], log_probs.shape[1], dtype=np.int64)
    if target_lengths is None:
        target_lengths = np.full(targets.shape[0], targets.shape[1], dtype=np.int64)

    paths, scores = forced_align_cpp(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
    )

    return paths, scores


import math

import math
import numpy as np
import torch

def get_alignments(
    emissions: torch.Tensor,
    tokens_list: list,
    tokenizer,
    desired_T: int = 1000,  # Desired sequence length per batch
):
    """
    Align tokens to emissions using forced alignment.

    Args:
        emissions (torch.Tensor): Emissions tensor of shape [T_total, C] or [B, T, C].
        tokens_list (List[List[str]]): List of token lists for each sequence in the batch.
        tokenizer: Tokenizer object with get_vocab(), pad_token_id, unk_token_id.
        desired_T (int, optional): Desired sequence length per batch. Defaults to 1000.

    Returns:
        Tuple[List, np.ndarray, str]: Segments list, scores array, and blank token.
    """
    assert len(tokens_list) > 0, "Empty tokens_list"
    assert desired_T > 0, "desired_T must be a positive integer"

    # Prepare the vocabulary
    dictionary = tokenizer.get_vocab()
    dictionary = {k.lower(): v for k, v in dictionary.items()}
    dictionary["<star>"] = len(dictionary)
    blank_id = dictionary.get("<blank>", tokenizer.pad_token_id)
    idx_to_token_map = {v: k for k, v in dictionary.items()}

    # Handle emissions tensor shape
    if emissions.dim() != 2:
        raise ValueError("Emissions tensor must have 2 dimensions [T_total, C].")

    T_total, C = emissions.size()
    device = emissions.device  # Get the device of emissions tensor

    # Automatically determine batch_size
    batch_size = math.ceil(T_total / desired_T)
    T = T_total // batch_size  # Integer division

    # Adjust T and batch_size if necessary
    remainder = T_total % batch_size
    if remainder != 0:
        T += 1  # Increase T to cover all time steps

    # Reshape emissions
    # Pad emissions if necessary
    padding = (batch_size * T) - T_total
    if padding > 0:
        padding_tensor = torch.zeros(padding, C, dtype=emissions.dtype, device=device)
        emissions = torch.cat([emissions, padding_tensor], dim=0)

    emissions = emissions.view(batch_size, T, C)  # Shape: (batch_size, T, C)

    # Prepare tokens_list
    # If tokens_list has fewer entries than batch_size, pad it with empty lists
    if len(tokens_list) < batch_size:
        tokens_list += [[] for _ in range(batch_size - len(tokens_list))]
    elif len(tokens_list) > batch_size:
        # Truncate tokens_list to match batch_size
        tokens_list = tokens_list[:batch_size]

    B = emissions.size(0)
    assert B == batch_size, "Emissions batch size must match computed batch_size"

    # Prepare targets
    max_L = max(len(tokens) for tokens in tokens_list)
    targets = np.full((B, max_L), fill_value=tokenizer.pad_token_id, dtype=np.int64)
    target_lengths = np.zeros(B, dtype=np.int64)

    for i, tokens in enumerate(tokens_list):
        token_indices = [
            dictionary.get(c.lower(), tokenizer.unk_token_id) for c in tokens
        ]
        L = len(token_indices)
        targets[i, :L] = token_indices
        target_lengths[i] = L

    # Convert emissions to numpy
    if emissions.is_cuda:
        emissions_np = emissions.cpu().float().numpy()
    else:
        emissions_np = emissions.float().numpy()

    # Input lengths
    input_lengths = np.full(B, T, dtype=np.int64)
    if padding > 0:
        input_lengths[-1] = T - padding  # Adjust length for last batch

    # Call the batch-enabled forced_align function
    paths, scores = forced_align(
        emissions_np,
        targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=blank_id,
    )

    # Process the outputs
    segments_list = []
    for i in range(B):
        path_length = input_lengths[i]
        path = paths[i, :path_length].tolist()
        segments = merge_repeats(path, idx_to_token_map)
        segments_list.append(segments)

    return segments_list, scores, idx_to_token_map[blank_id]





def load_alignment_model(
    device: str,
    model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
    attn_implementation: str = None,
    dtype: torch.dtype = torch.float32,
):
    if attn_implementation is None:
        if version.parse(transformers_version) < version.parse("4.41.0"):
            attn_implementation = "eager"
        elif (
            is_flash_attn_2_available()
            and device == "cuda"
            and dtype in [torch.float16, torch.bfloat16]
        ):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"

    model = (
        AutoModelForCTC.from_pretrained(
            model_path,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer
