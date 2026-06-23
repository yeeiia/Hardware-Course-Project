from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .config import save_config


METRIC_FIELDS = [
    "round",
    "phase",
    "dataset",
    "model",
    "split",
    "B",
    "E",
    "client_id",
    "train_loss",
    "global_loss",
    "accuracy",
    "macro_f1",
    "train_time",
    "eval_time",
    "bytes_sent",
    "bytes_recv",
    "samples",
    "peak_memory_mb",
    "status",
    "pi_temp",
    "pi_throttled",
]


class RunLogger:
    def __init__(
        self,
        config: dict[str, Any],
        early_stop_patience: int = 0,
        early_stop_min_delta: float = 0.001,
    ) -> None:
        root = Path(config["run"]["dir"])
        name = config["run"].get("name")
        if not name:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            name = f"{stamp}-{config['dataset']}-{config['partition']['type']}-b{config['batch_size']}-e{config['local_epochs']}"
        self.run_dir = root / name
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.figure_dir = self.run_dir / "figures"
        self._config_run = config["run"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, self.run_dir / "config.yaml")
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"
        self.records: list[dict[str, Any]] = []
        self.early_stop_patience = max(0, int(early_stop_patience))
        self.early_stop_min_delta = float(early_stop_min_delta)
        self.best_accuracy = float("-inf")
        self.best_round = 0
        self.rounds_without_improvement = 0
        self.jsonl_path.write_text("", encoding="utf-8")
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore").writeheader()

    def log(self, record: dict[str, Any]) -> None:
        normalized = {field: record.get(field, "") for field in METRIC_FIELDS}
        self.records.append(normalized)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore").writerow(normalized)

    def save_checkpoint(self, state_dict: dict[str, torch.Tensor], round_index: int) -> None:
        torch.save(state_dict, self.checkpoint_dir / f"global_round_{round_index:03d}.pt")

    def save_best(self, state_dict: dict[str, torch.Tensor], round_index: int, accuracy: float) -> None:
        if accuracy > self.best_accuracy + self.early_stop_min_delta:
            self.best_accuracy = float(accuracy)
            self.best_round = int(round_index)
            self.rounds_without_improvement = 0
            if self.config_run().get("save_best_model", False):
                torch.save(state_dict, self.checkpoint_dir / "best.pt")
        else:
            self.rounds_without_improvement += 1

    def check_early_stop(self) -> bool:
        return (
            self.early_stop_patience > 0
            and self.best_round > 0
            and self.rounds_without_improvement >= self.early_stop_patience
        )

    def config_run(self) -> dict[str, Any]:
        return self._config_run

    def plot_curves(self) -> None:
        eval_records = [r for r in self.records if r.get("phase") == "eval"]
        if not eval_records:
            return
        rounds = [int(r["round"]) for r in eval_records]
        loss = [float(r["global_loss"]) for r in eval_records]
        acc = [float(r["accuracy"]) for r in eval_records]

        plt.figure(figsize=(7, 4))
        plt.plot(rounds, loss, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Global loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figure_dir / "global_loss.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(rounds, acc, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figure_dir / "accuracy.png", dpi=150)
        plt.close()


def common_record(config: dict[str, Any], round_index: int, phase: str, client_id: str = "") -> dict[str, Any]:
    return {
        "round": round_index,
        "phase": phase,
        "dataset": config["dataset"],
        "model": config["model"],
        "split": config["partition"]["type"],
        "B": config["batch_size"],
        "E": config["local_epochs"],
        "client_id": client_id,
    }
