from __future__ import annotations

import argparse
import socket
from dataclasses import dataclass
from typing import Any

import torch

from .aggregator import fedavg
from .config import load_config
from .data import build_loader, load_data, seed_everything
from .evaluate import evaluate
from .metrics import RunLogger, common_record
from .models import build_model
from .protocol import Message, recv_message, send_message
from .serialization import bytes_to_state_dict, state_dict_to_bytes


@dataclass
class ClientConnection:
    client_id: str
    sock: socket.socket
    address: tuple[str, int]
    bytes_sent: int = 0
    bytes_recv: int = 0


def run_server(config: dict[str, Any]) -> str:
    seed_everything(int(config["seed"]))
    device = torch.device(config.get("device", "cpu"))
    data = load_data(config)
    eval_loader = build_loader(data.test, int(config["batch_size"]), shuffle=False, seed=int(config["seed"]))
    model = build_model(config["model"]).to(device)
    logger = RunLogger(config)

    server_cfg = config["server"]
    clients = _accept_clients(server_cfg, int(config["num_clients"]))
    try:
        for round_index in range(1, int(config["rounds"]) + 1):
            global_payload = state_dict_to_bytes(model.state_dict())
            for client in clients:
                sent = send_message(
                    client.sock,
                    "GLOBAL_MODEL",
                    {"round": round_index, "config": _client_visible_config(config)},
                    global_payload,
                )
                client.bytes_sent += sent

            results: list[tuple[ClientConnection, Message]] = []
            for client in clients:
                message = recv_message(client.sock)
                client.bytes_recv += message.raw_bytes
                if message.msg_type != "TRAIN_RESULT":
                    raise RuntimeError(f"{client.client_id} sent unexpected message {message.msg_type}")
                results.append((client, message))

            states = [bytes_to_state_dict(message.payload) for _, message in results]
            sample_counts = [int(message.metadata["samples"]) for _, message in results]
            aggregated = fedavg(states, sample_counts)
            model.load_state_dict(aggregated)

            for client, message in results:
                meta = message.metadata
                record = common_record(config, round_index, "train", client.client_id)
                record.update(
                    {
                        "train_loss": meta.get("train_loss", ""),
                        "train_time": meta.get("train_time", ""),
                        "samples": meta.get("samples", ""),
                        "bytes_sent": client.bytes_sent,
                        "bytes_recv": client.bytes_recv,
                        "status": meta.get("status", "ok"),
                    }
                )
                logger.log(record)

            eval_metrics = evaluate(model, eval_loader, device)
            eval_record = common_record(config, round_index, "eval", "server")
            eval_record.update(eval_metrics)
            eval_record.update({"bytes_sent": sum(c.bytes_sent for c in clients), "bytes_recv": sum(c.bytes_recv for c in clients)})
            logger.log(eval_record)
            if config["run"].get("save_every_round", True):
                logger.save_checkpoint(model.state_dict(), round_index)
            logger.plot_curves()
    finally:
        for client in clients:
            try:
                client.sock.close()
            except OSError:
                pass
    return str(logger.run_dir)


def _accept_clients(server_cfg: dict[str, Any], expected_clients: int) -> list[ClientConnection]:
    bind_host = str(server_cfg.get("bind_host") or server_cfg.get("host") or "0.0.0.0")
    port = int(server_cfg.get("port", 9000))
    timeout = float(server_cfg.get("timeout_seconds", 600))
    clients: list[ClientConnection] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bind_host, port))
        listener.listen(expected_clients)
        listener.settimeout(timeout)
        while len(clients) < expected_clients:
            sock, address = listener.accept()
            sock.settimeout(timeout)
            message = recv_message(sock)
            if message.msg_type != "REGISTER":
                send_message(sock, "ERROR", {"error": "first message must be REGISTER"})
                sock.close()
                continue
            client_id = str(message.metadata.get("client_id", f"client{len(clients)}"))
            clients.append(ClientConnection(client_id=client_id, sock=sock, address=address, bytes_recv=message.raw_bytes))
    min_clients = int(server_cfg.get("min_clients", expected_clients))
    if len(clients) < min_clients:
        raise RuntimeError(f"only {len(clients)} clients registered, need {min_clients}")
    return clients


def _client_visible_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": config["dataset"],
        "model": config["model"],
        "batch_size": config["batch_size"],
        "local_epochs": config["local_epochs"],
        "seed": config["seed"],
        "partition": config["partition"],
        "data": config["data"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FedAvg socket server")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_dir = run_server(load_config(args.config))
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
