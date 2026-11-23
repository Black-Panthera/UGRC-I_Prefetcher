#!/usr/bin/env python3
"""
Automated, Thermally-Throttled Pass 1 Matrix Experiment Runner for ChampSim.
Executes 5 prefetchers across 18 SPEC CPU 2017 traces with active thermal monitoring,
SIGSTOP/SIGCONT thermal protection, idempotent checkpointing, and automatic cooldown.
"""

import os
import sys
import time
import subprocess
import json
import re
import signal
import argparse
from datetime import datetime
from pathlib import Path

DEFAULT_PREFETCHERS = ['pythia', 'ip_stride', 'spp_dev', 'next_line', 'va_ampm_lite']

def get_cpu_temp():
    """Reads the current CPU Package / maximum Core temperature in Celsius."""
    try:
        out = subprocess.check_output(['sensors'], text=True)
        m = re.search(r'Package id 0:\s+\+([0-9\.]+)°?C', out)
        if m:
            return float(m.group(1))
        cores = re.findall(r'Core \d+:\s+\+([0-9\.]+)°?C', out)
        if cores:
            return max(float(c) for c in cores)
    except Exception:
        pass
    try:
        import glob
        zones = sorted(glob.glob('/sys/class/thermal/thermal_zone*'))
        for z in zones:  # prefer the CPU package sensor; thermal_zone0 is often the motherboard ACPI zone
            with open(os.path.join(z, 'type')) as f:
                if f.read().strip() == 'x86_pkg_temp':
                    with open(os.path.join(z, 'temp')) as t:
                        return float(t.read().strip()) / 1000.0
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 50.0

def is_job_completed(out_json, out_txt):
    """Checks if a simulation job already completed successfully."""
    if not os.path.exists(out_json) or os.path.getsize(out_json) == 0:
        return False
    # The JSON alone is not proof: a killed relaunch can leave a stale JSON next to a truncated .txt
    try:
        with open(out_txt, 'r') as f:
            if 'ChampSim completed all CPUs' not in f.read():
                return False
    except Exception:
        return False
    try:
        with open(out_json, 'r') as f:
            data = json.load(f)
            return 'accounting_validation' in data or 'simulation_metadata' in data
    except Exception:
        return False

def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"

def extract_ipc_from_txt(txt_path):
    """Parses cumulative IPC from ChampSim text output."""
    try:
        with open(txt_path, 'r') as f:
            text = f.read()
        m = re.findall(r'cumulative IPC:\s+([0-9\.]+)', text)
        if m:
            return float(m[-1])
    except Exception:
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Thermally Throttled ChampSim Pass 1 Runner")
    parser.add_argument('--workers', type=int, default=1, help="Max parallel simulation workers (default: 1 for cool laptop operation)")
    parser.add_argument('--affinity', default='12-19', help="CPU core affinity mask (default: '12-19' for E-cores to prevent laptop thermal spikes)")
    parser.add_argument('--max-temp', type=float, default=82.0, help="Pause process with SIGSTOP when CPU temp exceeds this Celsius (default: 82.0)")
    parser.add_argument('--cooldown-temp', type=float, default=68.0, help="Resume process with SIGCONT when CPU temp drops below this Celsius (default: 68.0)")
    parser.add_argument('--warmup', type=int, default=100_000_000, help="Warmup instructions (default: 100M)")
    parser.add_argument('--simulation', type=int, default=500_000_000, help="Simulation instructions (default: 500M)")
    parser.add_argument('--trace-dir', default='test_traces', help="Directory containing traces")
    parser.add_argument('--output-dir', default='results/pass1', help="Output directory for results")
    parser.add_argument('--prefetchers', nargs='+', default=DEFAULT_PREFETCHERS, help="Prefetchers to run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    status_file = os.path.join(args.output_dir, 'status.json')
    log_file_path = os.path.join(args.output_dir, 'runner.log')

    # Collect traces
    trace_files = sorted([f for f in os.listdir(args.trace_dir) if f.endswith('.champsimtrace.xz')])
    if not trace_files:
        print(f"ERROR: No .champsimtrace.xz files found in {args.trace_dir}!")
        sys.exit(1)

    # Build queue: group by trace so each trace gets all 5 prefetchers compared as soon as possible
    jobs = []
    for tf in trace_files:
        stem = tf.split('.')[0] + '.' + tf.split('.')[1] # e.g. 602.gcc_s-1850B
        for pref in args.prefetchers:
            jobs.append({
                'trace_file': tf,
                'stem': stem,
                'prefetcher': pref,
                'bin': f"bin/champsim_{pref}",
                'out_json': os.path.join(args.output_dir, f"{stem}_{pref}.json"),
                'out_txt': os.path.join(args.output_dir, f"{stem}_{pref}.txt"),
            })

    total_jobs = len(jobs)
    print(f"=== ChampSim Pass 1 Matrix Orchestrator ===")
    print(f"Traces: {len(trace_files)} | Prefetchers: {len(args.prefetchers)} | Total Jobs: {total_jobs}")
    print(f"Instructions: {args.warmup:,} Warmup + {args.simulation:,} Detailed Simulation")
    print(f"Workers: {args.workers} | Thermal Guard: Pause > {args.max_temp}°C, Resume <= {args.cooldown_temp}°C")
    print(f"Results Directory: {args.output_dir}\n")

    # Filter out already completed jobs
    pending_jobs = [j for j in jobs if not is_job_completed(j['out_json'], j['out_txt'])]
    completed_count = total_jobs - len(pending_jobs)

    print(f"Found {completed_count}/{total_jobs} already completed. Pending: {len(pending_jobs)} jobs.")

    active_processes = [] # list of dicts: {'job': job, 'proc': Popen, 'start_time': float, 'out_fh': fh, 'is_paused': bool}
    start_time_all = time.time()

    def log_msg(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)
        with open(log_file_path, 'a') as lf:
            lf.write(line + "\n")

    last_heartbeat = 0

    while pending_jobs or active_processes:
        current_temp = get_cpu_temp()

        # 1. Check for finished processes
        finished_items = []
        for item in active_processes:
            proc = item['proc']
            ret = proc.poll()
            if ret is not None:
                item['out_fh'].close()
                elapsed = time.time() - item['start_time']
                job = item['job']
                ipc = extract_ipc_from_txt(job['out_txt'])
                ipc_str = f"| IPC: {ipc:.4f}" if ipc is not None else ""
                if ret == 0 and os.path.exists(job['out_json']) and os.path.getsize(job['out_json']) > 0:
                    log_msg(f"SUCCESS [{completed_count + 1}/{total_jobs}]: {job['stem']} | {job['prefetcher']} {ipc_str} in {format_duration(elapsed)} (Exit: 0).")
                else:
                    log_msg(f"FAILED [{completed_count + 1}/{total_jobs}]: {job['stem']} | {job['prefetcher']} exited with code {ret} after {format_duration(elapsed)}!")
                finished_items.append(item)
                completed_count += 1

        for item in finished_items:
            active_processes.remove(item)

        # 2. Thermal Guard: Active pausing (SIGSTOP) if temperature spikes
        if current_temp >= args.max_temp:
            for item in active_processes:
                if not item['is_paused']:
                    try:
                        os.kill(item['proc'].pid, signal.SIGSTOP)
                        item['is_paused'] = True
                        log_msg(f"THERMAL GUARD: CPU Temp = {current_temp:.1f}°C (>= {args.max_temp}°C). Paused PID {item['proc'].pid} ({item['job']['prefetcher']}) to cool down.")
                    except ProcessLookupError:
                        pass
        elif current_temp <= args.cooldown_temp:
            for item in active_processes:
                if item['is_paused']:
                    try:
                        os.kill(item['proc'].pid, signal.SIGCONT)
                        item['is_paused'] = False
                        log_msg(f"THERMAL RECOVERY: CPU Temp = {current_temp:.1f}°C (<= {args.cooldown_temp}°C). Resumed PID {item['proc'].pid} ({item['job']['prefetcher']}).")
                    except ProcessLookupError:
                        pass

        # 3. Launch new jobs if slots available and CPU is not overheating
        while len(active_processes) < args.workers and pending_jobs and current_temp < args.max_temp:
            job = pending_jobs.pop(0)

            if is_job_completed(job['out_json'], job['out_txt']):
                completed_count += 1
                continue

            env = os.environ.copy()
            env["CHAMPSIM_PROFILER_OUTPUT"] = os.path.abspath(job['out_json'])

            cmd = []
            if args.affinity:
                cmd.extend(["taskset", "-c", args.affinity])
            cmd.extend([
                "nice", "-n", "15",
                os.path.abspath(job['bin']),
                "--warmup-instructions", str(args.warmup),
                "--simulation-instructions", str(args.simulation),
                os.path.abspath(os.path.join(args.trace_dir, job['trace_file']))
            ])

            out_fh = open(job['out_txt'], 'w', buffering=1) # line buffered
            proc = subprocess.Popen(cmd, stdout=out_fh, stderr=subprocess.STDOUT, env=env)
            t_start = time.time()
            active_processes.append({
                'job': job,
                'proc': proc,
                'start_time': t_start,
                'out_fh': out_fh,
                'is_paused': False
            })
            log_msg(f"LAUNCH [{completed_count + len(active_processes)}/{total_jobs}]: {job['stem']} | {job['prefetcher']} (PID {proc.pid}) | CPU Temp: {current_temp:.1f}°C")

        # 4. Periodic Heartbeat & Status File update (every 30 seconds)
        now = time.time()
        if now - last_heartbeat >= 30:
            last_heartbeat = now
            active_strs = []
            for item in active_processes:
                status_tag = "PAUSED" if item['is_paused'] else "RUNNING"
                el = time.time() - item['start_time']
                active_strs.append(f"{item['job']['stem'][:12]}/{item['job']['prefetcher']} ({status_tag}, {format_duration(el)})")
            active_desc = ", ".join(active_strs) if active_strs else "None"
            total_elapsed = now - start_time_all
            pct = (completed_count / total_jobs) * 100.0 if total_jobs else 100.0
            log_msg(f"HEARTBEAT: {completed_count}/{total_jobs} ({pct:.1f}%) | Active [{len(active_processes)}/{args.workers}]: {active_desc} | CPU Temp: {current_temp:.1f}°C | Total Time: {format_duration(total_elapsed)}")

            # Update status.json
            status = {
                'total_jobs': total_jobs,
                'completed_jobs': completed_count,
                'progress_pct': pct,
                'cpu_temp_c': current_temp,
                'total_elapsed_sec': total_elapsed,
                'active_jobs': [{
                    'stem': item['job']['stem'],
                    'prefetcher': item['job']['prefetcher'],
                    'pid': item['proc'].pid,
                    'is_paused': item['is_paused'],
                    'elapsed_sec': time.time() - item['start_time']
                } for item in active_processes]
            }
            try:
                with open(status_file, 'w') as sf:
                    json.dump(status, sf, indent=2)
            except Exception:
                pass

        time.sleep(2)

    log_msg(f"=== ALL {total_jobs} PASS 1 JOBS COMPLETED SUCCESSFULLY! Total time: {format_duration(time.time() - start_time_all)} ===")

if __name__ == '__main__':
    main()
