from __future__ import annotations

import io

import torch


def state_dict_to_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    cpu_state = {key: value.detach().cpu() for key, value in state_dict.items()}
    torch.save(cpu_state, buffer)
    return buffer.getvalue()


def bytes_to_state_dict(payload: bytes) -> dict[str, torch.Tensor]:
    buffer = io.BytesIO(payload)
    return torch.load(buffer, map_location="cpu")
