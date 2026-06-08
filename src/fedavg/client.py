from __future__ import annotations

import argparse
import socket
from typing import Any

import torch
from torch.utils.data import Subset

from .config import load_config
from .data import build_loader, load_data, seed_everything
from .models import build_model
from .partition import label_histogram, make_partitions
from .pi_status import read_pi_status
from .protocol import recv_message, send_message
from .serialization import bytes_to_state_dict, state_dict_to_bytes
from .train import train_local


def run_client(config: dict[str, Any], client_id: str, client_index: int | None = None) -> None:
    seed_everything(int(config["seed"]))
    index = _client_index(client_id, client_index)
    device = torch.device(config.get("device", "cpu"))
    data = load_data(config)
    partitions = make_partitions(data.labels, int(config["num_clients"]), config["partition"], int(config["seed"]))
    indices = partitions[index]
    subset = Subset(data.train, indices)
    loader = build_loader(subset, int(config["batch_size"]), shuffle=True, seed=int(config["seed"]) + index)
    histogram = label_histogram(data.labels, indices)
    model = build_model(config["model"]).to(device)

    server_cfg = config["server"]
    with socket.create_connection((str(server_cfg["host"]), int(server_cfg["port"])), timeout=float(server_cfg["timeout_seconds"])) as sock:
        sock.settimeout(float(server_cfg["timeout_seconds"]))
        send_message(sock, "REGISTER", {"client_id": client_id, "samples": len(indices), "label_histogram": histogram})
        while True:
            try:
                message = recv_message(sock)
            except EOFError:
                break
            if message.msg_type == "ERROR":
                raise RuntimeError(str(message.metadata))
            if message.msg_type != "GLOBAL_MODEL":
                raise RuntimeError(f"unexpected message type: {message.msg_type}")

            round_index = int(message.metadata["round"])
            state = bytes_to_state_dict(message.payload)
            model.load_state_dict(state)
            stats = train_local(model, loader, config, device)
            payload = state_dict_to_bytes(model.state_dict())
            metadata: dict[str, Any] = {
                "client_id": client_id,
                "round": round_index,
                "samples": int(stats["samples"]),
                "train_loss": stats["train_loss"],
                "train_time": stats["train_time"],
                "status": "ok",
            }
            metadata.update(read_pi_status())
            send_message(sock, "TRAIN_RESULT", metadata, payload)


def _client_index(client_id: str, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    digits = "".join(ch for ch in client_id if ch.isdigit())
    if digits:
        return int(digits)
    raise ValueError("client index is required when client_id has no numeric suffix")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FedAvg socket client")
    parser.add_argument("--config", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-index", type=int, default=None)
    args = parser.parse_args(argv)
    run_client(load_config(args.config), args.client_id, args.client_index)


if __name__ == "__main__":
    main()
