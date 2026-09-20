#!/usr/bin/env python3
"""
Pathfinder SNN Prediction Generator for ChampSim.

Consumes recorded L2C/LLC demand load miss files from `pathfinder_recorder`
and runs the authentic PyTorch/BindsNET Spiking Neural Network (SNN) model
to learn delta patterns per (PC, page) and predict lookahead prefetch addresses.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

# Plumbing to ensure local bindsnet and model modules are imported correctly
ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR / "bindsnet"))
sys.path.insert(0, str(ENGINE_DIR))

import model


def load_record_data(record_path: Path) -> List[Tuple[int, int, int, int, bool]]:
    data = []
    if not record_path.exists():
        raise FileNotFoundError(f"Record file not found: {record_path}")

    with open(record_path, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                # (instr_id, cycle_count, load_addr, load_ip, llc_hit)
                instr_id = int(parts[0])
                cycle = int(parts[1])
                addr = int(parts[2], 16)
                ip = int(parts[3], 16)
                hit = (int(parts[4]) == 1)
                data.append((instr_id, cycle, addr, ip, hit))
    return data


def generate_trace_predictions(record_file: str, output_file: str) -> int:
    rec_path = Path(record_file)
    out_path = Path(output_file)

    print(f"\n[SNN Engine] Processing: {rec_path.name}")
    start_t = time.time()
    data = load_record_data(rec_path)
    print(f"[SNN Engine] Loaded {len(data)} recorded entries from {rec_path.name}")

    if not data:
        print(f"[SNN Engine] Warning: No data in {rec_path.name}")
        out_path.write_text("")
        return 0

    snn = model.MLPrefetchModel_pathfinder_pcpage()
    predictions = snn.generate(data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for instr_id, pf_addr in predictions:
            f.write(f"{instr_id} {hex(pf_addr)[2:]}\n")

    elapsed = time.time() - start_t
    print(
        f"[SNN Engine] Finished {rec_path.name}: Generated {len(predictions)} "
        f"lookahead prefetch predictions in {elapsed:.1f}s -> {out_path.name}"
    )
    return len(predictions)


def process_wrapper(args_tuple):
    rec_f, out_f = args_tuple
    return generate_trace_predictions(rec_f, out_f)


def main():
    parser = argparse.ArgumentParser(description="Run Pathfinder SNN prediction generator.")
    parser.add_argument("--record-file", help="Input recorded trace file (.txt)")
    parser.add_argument("--output-file", help="Output predictions file (.txt)")
    parser.add_argument("--all", action="store_true", help="Generate predictions for all 4 standard benchmark traces in parallel")
    parser.add_argument("--champsim-root", default="/home/rama/ChampSim", help="ChampSim repository root directory")

    args = parser.parse_args()
    root = Path(args.champsim_root)

    if args.all:
        traces = [
            "600.perlbench_s-210B",
            "602.gcc_s-1850B",
            "605.mcf_s-1152B",
            "619.lbm_s-2676B",
        ]
        tasks = []
        for tr in traces:
            rec = root / f"pathfinder_record_{tr}.txt"
            out = root / f"pathfinder_predictions_{tr}.txt"
            tasks.append((str(rec), str(out)))

        print(f"Launching SNN prediction generation across {len(tasks)} traces in parallel...")
        # Use available CPU cores for concurrent trace processing
        max_workers = min(len(tasks), os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(process_wrapper, tasks))

        print(f"\nAll SNN prediction files successfully generated! Total predictions: {sum(results)}")
    elif args.record_file and args.output_file:
        generate_trace_predictions(args.record_file, args.output_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

