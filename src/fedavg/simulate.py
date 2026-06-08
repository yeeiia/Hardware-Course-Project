from __future__ import annotations

import argparse
import multiprocessing as mp
import time

from .client import run_client
from .config import load_config
from .server import run_server


def _server_worker(config: dict) -> None:
    run_dir = run_server(config)
    print(f"SERVER_RUN_DIR={run_dir}", flush=True)


def _client_worker(config: dict, client_id: str, client_index: int) -> None:
    run_client(config, client_id, client_index)


def run_simulation(config: dict, clients: int) -> None:
    config["num_clients"] = clients
    config["server"]["host"] = "127.0.0.1"
    config["server"]["bind_host"] = "127.0.0.1"
    config["server"]["min_clients"] = clients

    ctx = mp.get_context("spawn")
    server_proc = ctx.Process(target=_server_worker, args=(config,), name="fedavg-server")
    server_proc.start()
    time.sleep(2.0)

    client_procs = [
        ctx.Process(target=_client_worker, args=(config, f"client{i}", i), name=f"fedavg-client-{i}")
        for i in range(clients)
    ]
    for proc in client_procs:
        proc.start()

    for proc in client_procs:
        proc.join()
    server_proc.join()

    failures = {proc.name: proc.exitcode for proc in [server_proc, *client_procs] if proc.exitcode != 0}
    if failures:
        raise SystemExit(f"simulation failed: {failures}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a local multi-process FedAvg simulation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--clients", type=int, default=2)
    args = parser.parse_args(argv)
    run_simulation(load_config(args.config), args.clients)


if __name__ == "__main__":
    mp.freeze_support()
    main()
