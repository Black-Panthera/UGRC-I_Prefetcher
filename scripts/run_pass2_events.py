#!/usr/bin/env python3
"""
Pass-2 runner: full-length runs (default 100M warmup + 200M measured, the same region as pass 1;
Ligra traces are only 250M instructions long, so they get 100M + 150M as in the Pythia paper)
with the L2C event logs enabled (CHAMPSIM_EVENT_LOG). Jobs are queued trace by trace, all
prefetchers of one trace before the next trace. Each run's .acc.bin/.pf.bin are zstd-compressed
as soon as the run finishes, and a job only counts as done once its .txt completed and both
compressed logs exist.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pass1_matrix import get_cpu_temp, format_duration, extract_ipc_from_txt  # noqa: E402

CORE_TRACES = ['623.xalancbmk_s-10B', '620.omnetpp_s-141B', '620.omnetpp_s-874B', '627.cam4_s-573B',
               '623.xalancbmk_s-592B', '605.mcf_s-994B']
CONTRAST_TRACES = ['649.fotonik3d_s-10881B', '619.lbm_s-2677B', '607.cactuBSSN_s-2421B']
# Traces where Pythia beat the best of SPP/Bingo/MLOP by a wide margin in the authors' MICRO'21 data (test_traces/pythia_strong)
STRONG_TRACES = ['429.mcf-184B', '429.mcf-192B', '429.mcf-51B', '450.soplex-247B', '437.leslie3d-149B', '437.leslie3d-232B',
                 '437.leslie3d-265B', '437.leslie3d-271B', '482.sphinx3-1297B', '433.milc-127B'] + [
    f'ligra_{app}.com-lj.ungraph.gcc_6.3.0_O3.drop_{drop}.length_250M' for app, drop in [
        ('BC', '26750M'), ('PageRankDelta', '24000M'), ('PageRankDelta', '24500M'), ('Radii', '32000M'), ('BFS-Bitvector', '23000M'),
        ('Triangle', '25000M'), ('PageRank', '51000M'), ('Components-Shortcut', '22000M'), ('Components', '22750M')]]
LIGRA_SIMULATION = 150_000_000
PREFETCHERS = ['pythia', 'spp_dev', 'ip_stride', 'next_line', 'va_ampm_lite', 'no']


def txt_complete(path):
    try:
        with open(path) as f:
            return 'ChampSim completed all CPUs' in f.read()
    except OSError:
        return False


def job_done(job):
    return txt_complete(job['txt']) and all(os.path.exists(job['prefix'] + ext + '.zst') for ext in ('.acc.bin', '.pf.bin'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch', choices=['core', 'contrast', 'all', 'strong'], default='all')
    parser.add_argument('--traces', nargs='+', help='Override the trace list (stems)')
    parser.add_argument('--prefetchers', nargs='+', default=PREFETCHERS)
    parser.add_argument('--workers', type=int, default=18)
    parser.add_argument('--affinity', default=f'0-{os.cpu_count() - 1}')
    parser.add_argument('--max-temp', type=float, default=98.0)
    parser.add_argument('--cooldown-temp', type=float, default=94.0)
    parser.add_argument('--warmup', type=int, default=100_000_000)
    parser.add_argument('--simulation', type=int, default=200_000_000,
                        help=f'measured instructions; Ligra traces (250M long) use {LIGRA_SIMULATION:,} unless --ligra-simulation is given')
    parser.add_argument('--ligra-simulation', type=int, default=LIGRA_SIMULATION)
    parser.add_argument('--trace-dir', default=None, help='default: test_traces, or test_traces/pythia_strong for --batch strong')
    parser.add_argument('--output-dir', default=None, help='default: results/pass2_events, or results/pass2_strong for --batch strong')
    args = parser.parse_args()

    strong = args.batch == 'strong'
    args.trace_dir = args.trace_dir or ('test_traces/pythia_strong' if strong else 'test_traces')
    args.output_dir = args.output_dir or ('results/pass2_strong' if strong else 'results/pass2_events')
    traces = args.traces or {'core': CORE_TRACES, 'contrast': CONTRAST_TRACES, 'all': CORE_TRACES + CONTRAST_TRACES,
                             'strong': STRONG_TRACES}[args.batch]
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'runner.log')

    def log(msg):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    jobs = []
    for stem in traces:
        for pref in args.prefetchers:
            base = os.path.join(args.output_dir, f'{stem}_{pref}')
            jobs.append({'stem': stem, 'pref': pref, 'prefix': os.path.abspath(base), 'txt': base + '.txt', 'json': base + '.json',
                         'trace': os.path.abspath(os.path.join(args.trace_dir, f'{stem}.champsimtrace.xz')),
                         'bin': os.path.abspath(f'bin/champsim_{pref}'),
                         'simulation': args.ligra_simulation if stem.startswith('ligra_') else args.simulation})
    for j in jobs:
        if not os.path.exists(j['trace']) or not os.path.exists(j['bin']):
            sys.exit(f"missing trace or binary for {j['stem']} / {j['pref']}")

    pending = [j for j in jobs if not job_done(j)]
    log(f"=== Pass 2 event-log runner: {len(traces)} traces x {len(args.prefetchers)} configs = {len(jobs)} jobs, "
        f"{len(jobs) - len(pending)} already done ===")
    log(f"Instructions: {args.warmup:,} warmup + {args.simulation:,} measured (Ligra: {args.ligra_simulation:,}) | "
        f"workers {args.workers} | affinity {args.affinity} | pause >= {args.max_temp}C, resume <= {args.cooldown_temp}C")

    running, compressing = [], []
    start_all = time.time()
    last_status = 0
    failed = 0

    while pending or running or compressing:
        temp = get_cpu_temp()

        for item in list(running):
            ret = item['proc'].poll()
            if ret is None:
                continue
            item['fh'].close()
            running.remove(item)
            job = item['job']
            elapsed = format_duration(time.time() - item['start'])
            if ret == 0 and txt_complete(job['txt']):
                ipc = extract_ipc_from_txt(job['txt'])
                log(f"SIM DONE: {job['stem']} | {job['pref']} | IPC {ipc} in {elapsed}; compressing logs")
                for ext in ('.acc.bin', '.pf.bin'):
                    raw = job['prefix'] + ext
                    proc = subprocess.Popen(['nice', '-n', '10', 'zstd', '-q', '-T2', '-f', '--rm', raw, '-o', raw + '.zst'])
                    compressing.append({'job': job, 'proc': proc, 'file': raw})
            else:
                failed += 1
                log(f"FAILED: {job['stem']} | {job['pref']} exit {ret} after {elapsed}")

        for item in list(compressing):
            ret = item['proc'].poll()
            if ret is None:
                continue
            compressing.remove(item)
            if ret != 0:
                failed += 1
                log(f"COMPRESS FAILED: {item['file']} (exit {ret})")
            elif job_done(item['job']) and not any(c['job'] is item['job'] for c in compressing):
                log(f"JOB COMPLETE: {item['job']['stem']} | {item['job']['pref']}")

        if temp >= args.max_temp:
            for item in running:
                if not item['paused']:
                    os.kill(item['proc'].pid, signal.SIGSTOP)
                    item['paused'] = True
        elif temp <= args.cooldown_temp:
            for item in running:
                if item['paused']:
                    os.kill(item['proc'].pid, signal.SIGCONT)
                    item['paused'] = False

        while pending and len(running) < args.workers and temp < args.max_temp:
            job = pending.pop(0)
            env = dict(os.environ, CHAMPSIM_PROFILER_OUTPUT=os.path.abspath(job['json']), CHAMPSIM_EVENT_LOG=job['prefix'])
            cmd = ['taskset', '-c', args.affinity, 'nice', '-n', '15', job['bin'],
                   '--warmup-instructions', str(args.warmup), '--simulation-instructions', str(job['simulation']), job['trace']]
            fh = open(job['txt'], 'w')
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running.append({'job': job, 'proc': proc, 'fh': fh, 'start': time.time(), 'paused': False})
            log(f"LAUNCH: {job['stem']} | {job['pref']} | {args.warmup // 10**6}M + {job['simulation'] // 10**6}M (PID {proc.pid}) | {temp:.0f}C")

        now = time.time()
        if now - last_status >= 60:
            last_status = now
            done = sum(job_done(j) for j in jobs)
            status = {'done': done, 'total': len(jobs), 'failed': failed, 'running': len(running), 'compressing': len(compressing),
                      'paused': sum(i['paused'] for i in running), 'temp_c': temp, 'elapsed': format_duration(now - start_all),
                      'active': [f"{i['job']['stem']}|{i['job']['pref']}" for i in running]}
            with open(os.path.join(args.output_dir, 'status.json'), 'w') as f:
                json.dump(status, f, indent=2)
        time.sleep(2)

    done = sum(job_done(j) for j in jobs)
    log(f"=== FINISHED: {done}/{len(jobs)} jobs complete, {failed} failures, total {format_duration(time.time() - start_all)} ===")


if __name__ == '__main__':
    main()
