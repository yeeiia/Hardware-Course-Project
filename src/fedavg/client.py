"""
FedAvg 客户端 / 训练方（在 PC 模拟时是子进程，在 Pi 上是单独主机）：

整体职责（一个进程跑完一整次实验）：
  1. 加载并按种子复现地切出"自己这一份"本地训练数据
  2. 连服务器，发 REGISTER 自报身份+样本数+标签直方图
  3. 进入循环：收 GLOBAL_MODEL → 本地训 E 个 epoch → 回 TRAIN_RESULT
     直到服务器关闭连接（recv 抛 EOFError 退出）

为什么客户端需要标签直方图？
  服务器侧分析 non-IID 程度时要看每个客户端的标签分布；
  这个信息在 REGISTER 阶段一次性传给服务器即可。
"""

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
    """客户端主流程：建本地数据 → 连服务器 → 多轮 train-and-upload。"""
    # 与服务器同一种子：保证两边切出的 partitions 一致，本客户端拿到正确的那一份。
    seed_everything(int(config["seed"]))
    # client_index 决定本进程取哪一份分区（"我是 0 号还是 1 号客户端？"）。
    index = _client_index(client_id, client_index)
    device = torch.device(config.get("device", "cpu"))

    # 关键设计：每个客户端"本地"也加载完整数据集，再用同样的 seed+num_clients
    # 复现服务器侧的切分。这样 partition 信息根本不需要通过网络传，省事且对齐。
    # 在真实联邦场景下数据天然只在本地，这里是仿真所以走"复现切分"的捷径。
    data = load_data(config)
    partitions = make_partitions(data.labels, int(config["num_clients"]), config["partition"], int(config["seed"]))
    indices = partitions[index]
    # Subset 是 PyTorch 提供的"按索引取子集"包装，不会复制数据，便于多客户端共享底层数据集。
    subset = Subset(data.train, indices)
    # 不同客户端用不同 shuffle 种子 (seed + index)，避免 N 个进程画一样的 mini-batch 序列。
    loader = build_loader(subset, int(config["batch_size"]), shuffle=True, seed=int(config["seed"]) + index)
    # 报给服务器看的标签分布；用来分析 non-IID。
    histogram = label_histogram(data.labels, indices)
    # 模型只创建结构；权重等着 GLOBAL_MODEL 来覆盖。
    model = build_model(config["model"]).to(device)

    server_cfg = config["server"]
    # create_connection 是 connect 的高级封装：解析地址 + 建连 + 设超时。
    with socket.create_connection((str(server_cfg["host"]), int(server_cfg["port"])), timeout=float(server_cfg["timeout_seconds"])) as sock:
        sock.settimeout(float(server_cfg["timeout_seconds"]))
        # === 握手：第一帧必须是 REGISTER（服务器 _accept_clients 强制要求）===
        send_message(sock, "REGISTER", {"client_id": client_id, "samples": len(indices), "label_histogram": histogram})

        # === 主循环：每轮被动地等服务器下发新模型，训完回传 ===
        while True:
            try:
                message = recv_message(sock)
            except EOFError:
                # 服务器把连接关了 = 训练结束的正常退出信号。
                break
            if message.msg_type == "ERROR":
                # 服务器主动报错，把消息原样抛出便于排查。
                raise RuntimeError(str(message.metadata))
            if message.msg_type != "GLOBAL_MODEL":
                # 当前协议中客户端只可能收到 GLOBAL_MODEL（或 ERROR）。
                raise RuntimeError(f"unexpected message type: {message.msg_type}")

            # --- ① 收下全局模型，覆盖本地权重 ---
            # 每轮都重置本地权重——这是 FedAvg 的关键：本地是"在全局基础上跑 E epoch"，
            # 而不是"在自己上一轮的本地权重上继续训"。
            round_index = int(message.metadata["round"])
            state = bytes_to_state_dict(message.payload)
            model.load_state_dict(state)

            # --- ② 本地训练 E 个 epoch (具体见 train.py)
            stats = train_local(model, loader, config, device)

            # --- ③ 把训练后的权重序列化回传 ---
            payload = state_dict_to_bytes(model.state_dict())
            metadata: dict[str, Any] = {
                "client_id": client_id,
                "round": round_index,                      # 与下发轮次对齐
                "samples": int(stats["samples"]),          # 给 fedavg 当加权系数 n_k
                "train_loss": stats["train_loss"],
                "train_time": stats["train_time"],
                "status": "ok",
            }
            # Pi 专用：附带温度 / 是否被降频，便于服务器端关联硬件状况。
            # 在 PC 上 read_pi_status 会返回空 dict，无副作用。
            metadata.update(read_pi_status())
            send_message(sock, "TRAIN_RESULT", metadata, payload)


def _client_index(client_id: str, explicit: int | None) -> int:
    """决定本客户端在 partitions 列表中取第几份。

    优先用显式传进来的 --client-index；否则从 client_id 里抠数字 (pi0→0, client1→1)。
    """
    if explicit is not None:
        return int(explicit)
    digits = "".join(ch for ch in client_id if ch.isdigit())
    if digits:
        return int(digits)
    raise ValueError("client index is required when client_id has no numeric suffix")


def main(argv: list[str] | None = None) -> None:
    """命令行入口：python -m fedavg.client --config xxx --client-id pi0 --client-index 0"""
    parser = argparse.ArgumentParser(description="FedAvg socket client")
    parser.add_argument("--config", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-index", type=int, default=None)
    args = parser.parse_args(argv)
    run_client(load_config(args.config), args.client_id, args.client_index)


if __name__ == "__main__":
    main()
