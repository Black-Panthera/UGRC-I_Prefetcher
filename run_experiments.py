#!/usr/bin/env python3
import json
import os
import subprocess
import sys

CHAMPSIM_ROOT = "/home/rama/ChampSim"
RESULTS_DIR = os.path.join(CHAMPSIM_ROOT, "results")
TRACES_DIR = os.path.join(CHAMPSIM_ROOT, "traces")

TRACES = [
    ("600.perlbench_s-210B", os.path.join(TRACES_DIR, "600.perlbench_s-210B.champsimtrace.xz")),
    ("602.gcc_s-1850B", os.path.join(TRACES_DIR, "602.gcc_s-1850B.champsimtrace.xz")),
    ("605.mcf_s-1152B", os.path.join(TRACES_DIR, "605.mcf_s-1152B.champsimtrace.xz")),
    ("619.lbm_s-2676B", os.path.join(TRACES_DIR, "619.lbm_s-2676B.champsimtrace.xz")),
]

WARMUP = 100000
SIM = 1000000

def run_cmd(cmd, env=None):
    print(f"[RUN] {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=CHAMPSIM_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed with return code {res.returncode}:\n{res.stdout}")
        sys.exit(1)
    return res.stdout

def build_binary(config_dict, config_path):
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    run_cmd(f"./config.sh {config_path}")
    run_cmd(f"make -j$(nproc)")

def get_base_config():
    with open(os.path.join(CHAMPSIM_ROOT, "champsim_config.json"), "r") as f:
        return json.load(f)

def run_pythia():
    print("\n" + "="*60 + "\nRUNNING PYTHIA EXPERIMENTS\n" + "="*60)
    for addr_mode, virt_pref in [("physical", False), ("virtual", True)]:
        cfg = get_base_config()
        cfg["executable_name"] = f"champsim_pythia_{addr_mode}"
        cfg["L1D"]["prefetcher"] = "no"
        cfg["L2C"]["prefetcher"] = "pythia"
        cfg["L2C"]["virtual_prefetch"] = virt_pref
        cfg["LLC"]["prefetcher"] = "no"
        
        cfg_path = os.path.join(CHAMPSIM_ROOT, f"config_pythia_{addr_mode}.json")
        print(f"\nBuilding Pythia ({addr_mode})...")
        build_binary(cfg, cfg_path)
        
        bin_path = os.path.join(CHAMPSIM_ROOT, "bin", f"champsim_pythia_{addr_mode}")
        
        for trace_name, trace_file in TRACES:
            out_name = f"{trace_name}_L2C_pythia_{addr_mode}_1core_100Kwarm_1Msim.txt"
            out_path = os.path.join(RESULTS_DIR, out_name)
            print(f"\nRunning Pythia {addr_mode} on {trace_name} -> {out_name}...")
            sim_cmd = f"{bin_path} --warmup-instructions {WARMUP} --simulation-instructions {SIM} {trace_file}"
            out = run_cmd(sim_cmd)
            with open(out_path, "w") as out_f:
                out_f.write(out)
            print(f"[DONE] Saved {out_path}")

def run_pathfinder():
    print("\n" + "="*60 + "\nRUNNING PATHFINDER EXPERIMENTS\n" + "="*60)
    
    # 1. Build L2C recorder
    cfg_rec = get_base_config()
    cfg_rec["executable_name"] = "champsim_pathfinder_record"
    cfg_rec["L1D"]["prefetcher"] = "no"
    cfg_rec["L2C"]["prefetcher"] = "pathfinder_recorder"
    cfg_rec["LLC"]["prefetcher"] = "no"
    cfg_rec_path = os.path.join(CHAMPSIM_ROOT, "config_pathfinder_record.json")
    print("\nBuilding Pathfinder Recorder (L2C)...")
    build_binary(cfg_rec, cfg_rec_path)
    bin_rec = os.path.join(CHAMPSIM_ROOT, "bin", "champsim_pathfinder_record")
    
    # Record L2C accesses and generate predictions for each trace
    pred_files = {}
    for trace_name, trace_file in TRACES:
        rec_file = os.path.join(CHAMPSIM_ROOT, f"pathfinder_record_{trace_name}.txt")
        pred_file = os.path.join(CHAMPSIM_ROOT, f"pathfinder_predictions_{trace_name}.txt")
        pred_files[trace_name] = pred_file
        
        print(f"\nRecording L2C accesses for {trace_name}...")
        env = os.environ.copy()
        env["PATHFINDER_RECORD_FILE"] = rec_file
        sim_cmd = f"{bin_rec} --warmup-instructions {WARMUP} --simulation-instructions {SIM} {trace_file}"
        run_cmd(sim_cmd, env=env)
        
    # Run SNN model in pathfinder_engine to generate authentic lookahead predictions
    python_snn = "/home/rama/miniconda3/envs/snn-champ_test/bin/python"
    if not os.path.exists(python_snn):
        python_snn = sys.executable

    print("\nExecuting Pathfinder SNN Model to generate lookahead predictions...")
    snn_script = os.path.join(CHAMPSIM_ROOT, "pathfinder_engine", "generate_predictions.py")
    run_cmd(f"{python_snn} {snn_script} --all")
    print("SNN Predictions generated for all traces.")
    
    # 2. Build and run pathfinder_file at L2C
    for addr_mode, virt_pref in [("physical", False), ("virtual", True)]:
        cfg_play = get_base_config()
        cfg_play["executable_name"] = f"champsim_pathfinder_{addr_mode}"
        cfg_play["L1D"]["prefetcher"] = "no"
        cfg_play["L2C"]["prefetcher"] = "pathfinder_file"
        cfg_play["L2C"]["virtual_prefetch"] = virt_pref
        cfg_play["LLC"]["prefetcher"] = "no"
        
        cfg_play_path = os.path.join(CHAMPSIM_ROOT, f"config_pathfinder_{addr_mode}.json")
        print(f"\nBuilding Pathfinder Replay ({addr_mode})...")
        build_binary(cfg_play, cfg_play_path)
        
        bin_play = os.path.join(CHAMPSIM_ROOT, "bin", f"champsim_pathfinder_{addr_mode}")
        
        for trace_name, trace_file in TRACES:
            out_name = f"{trace_name}_L2C_pathfinder_{addr_mode}_1core_100Kwarm_1Msim.txt"
            out_path = os.path.join(RESULTS_DIR, out_name)
            print(f"\nRunning Pathfinder {addr_mode} on {trace_name} -> {out_name}...")
            env = os.environ.copy()
            env["PATHFINDER_PREDICTIONS_FILE"] = pred_files[trace_name]
            sim_cmd = f"{bin_play} --warmup-instructions {WARMUP} --simulation-instructions {SIM} {trace_file}"
            out = run_cmd(sim_cmd, env=env)
            with open(out_path, "w") as out_f:
                out_f.write(out)
            print(f"[DONE] Saved {out_path}")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_pythia()
    run_pathfinder()
    print("\n" + "="*60 + "\nALL EXPERIMENTS COMPLETED SUCCESSFULLY!\n" + "="*60)

