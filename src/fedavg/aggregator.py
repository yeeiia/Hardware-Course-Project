from __future__ import annotations

import torch


def fedavg(state_dicts: list[dict[str, torch.Tensor]], sample_counts: list[int]) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("no state dictionaries provided")
    if len(state_dicts) != len(sample_counts):
        raise ValueError("state_dicts and sample_counts length mismatch")
    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("sample count total must be positive")

    result: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        accum = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
        for state, count in zip(state_dicts, sample_counts):
            accum += state[key].detach().cpu().float() * (float(count) / total)
        target_dtype = state_dicts[0][key].dtype
        if torch.is_floating_point(state_dicts[0][key]):
            result[key] = accum.to(dtype=target_dtype)
        else:
            result[key] = accum.round().to(dtype=target_dtype)
    return result
