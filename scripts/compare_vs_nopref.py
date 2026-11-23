#!/usr/bin/env python3
"""
Compares every prefetcher run in a pass-1 results directory against the matching
no-prefetcher run ({stem}_no.txt/.json) and

  1. writes summary_vs_nopref.csv with speedup, coverage and overprediction
     (same definitions as the Pythia MICRO'21 artifact, Appendix A.6):
        speedup        = IPC_X / IPC_nopref
        coverage       = (LLC_load_miss_nopref - LLC_load_miss_X) / LLC_load_miss_nopref
        overprediction = (LLC_read_miss_X - LLC_read_miss_nopref) / LLC_read_miss_nopref
                         where LLC_read_miss = LLC LOAD + PREFETCH misses minus MSHR merges
                         (duplicate requests for a line already in flight do not reach DRAM;
                         without this, SPP's duplicate prefetches look like ~170% overprediction)
  2. fills the two profiler fields that cannot be computed inside a single run:
        net_demand_miss_increase_vs_no_prefetch  (L2C data-demand misses, X - nopref; negative = fewer misses)
        dram_bandwidth_traffic_increase_pct      (DRAM reads+writes, % change vs nopref)
  3. optionally prints the authors' per-trace speedups next to ours (--authors-csv).
"""

import argparse
import csv
import glob
import json
import math
import os
import re

PREFETCHERS = ['pythia', 'spp_dev', 'bingo', 'mlop', 'ip_stride', 'next_line', 'va_ampm_lite']
AUTHORS_EXP = {'pythia': 'pythia', 'spp_dev': 'spp', 'bingo': 'bingo', 'mlop': 'mlop'}


def parse_txt(path):
    """Returns ROI stats from a ChampSim text output, or None if missing/incomplete."""
    if not os.path.exists(path):
        return None
    text = open(path).read()
    if 'ChampSim completed all CPUs' not in text:
        return None
    text = text[text.find('Region of Interest Statistics'):]

    def miss(cache, kind, unique=False):
        # MISS includes MSHR merges (requests for a line already in flight); unique=True drops them
        m = re.search(rf'cpu0->{cache}\s+{kind}\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)\s+MISS_MERGE:\s+(\d+)', text)
        if not m:
            return None
        return int(m.group(1)) - int(m.group(2)) if unique else int(m.group(1))

    def dram(queue):
        hit = re.search(rf'Channel 0 {queue} ROW_BUFFER_HIT:\s+(\d+)\s+ROW_BUFFER_MISS:\s+(\d+)', text)
        return int(hit.group(1)) + int(hit.group(2)) if hit else None

    ipc = re.search(r'CPU 0 cumulative IPC: ([\d.]+)', text)
    return {
        'ipc': float(ipc.group(1)),
        'llc_load_miss': miss('LLC', 'LOAD'),
        'llc_load_miss_unique': miss('LLC', 'LOAD', unique=True),
        'llc_pf_miss_unique': miss('LLC', 'PREFETCH', unique=True),
        'dram_traffic': (dram('RQ') or 0) + (dram('WQ') or 0),
    }


def runner_log_ipc(results_dir):
    """IPC per (stem, prefetcher) from runner.log; fallback for runs whose .txt was lost."""
    ipcs = {}
    log = os.path.join(results_dir, 'runner.log')
    if os.path.exists(log):
        for m in re.finditer(r'SUCCESS \[\d+/\d+\]: (\S+) \| (\S+) \| IPC: ([\d.]+)', open(log).read()):
            ipcs[(m.group(1), m.group(2))] = float(m.group(3))
    return ipcs


def geomean(values):
    values = [v for v in values if v]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', default='results/pass1')
    parser.add_argument('--authors-csv', default='results/reference/pythia_micro21_1C_base_config.csv')
    parser.add_argument('--no-json-update', action='store_true', help="Only write the CSV, do not touch the JSON files")
    args = parser.parse_args()

    log_ipc = runner_log_ipc(args.results_dir)
    stems = sorted(os.path.basename(f)[:-len('_no.txt')] for f in glob.glob(os.path.join(args.results_dir, '*_no.txt')))

    authors = {}
    if os.path.exists(args.authors_csv):
        for r in csv.DictReader(open(args.authors_csv)):
            authors.setdefault(r['Trace'], {})[r['Exp']] = float(r['Core_0_IPC'])

    rows = []
    for stem in stems:
        base_txt = parse_txt(os.path.join(args.results_dir, f'{stem}_no.txt'))
        base_json_path = os.path.join(args.results_dir, f'{stem}_no.json')
        if base_txt is None or not os.path.exists(base_json_path):
            print(f'skip {stem}: no-prefetch run missing or incomplete')
            continue
        base_json = json.load(open(base_json_path))
        base_l2_demand_miss = base_json['accounting_validation']['l2c_data_demand_misses']
        base_read_miss = base_txt['llc_load_miss_unique'] + base_txt['llc_pf_miss_unique']

        for pref in PREFETCHERS:
            json_path = os.path.join(args.results_dir, f'{stem}_{pref}.json')
            if not os.path.exists(json_path):
                continue
            x_txt = parse_txt(os.path.join(args.results_dir, f'{stem}_{pref}.txt'))
            ipc = x_txt['ipc'] if x_txt else log_ipc.get((stem, pref))
            row = {'trace': stem, 'prefetcher': pref, 'ipc': ipc, 'ipc_nopref': base_txt['ipc'],
                   'speedup': ipc / base_txt['ipc'] if ipc else None,
                   'coverage': None, 'overprediction': None, 'dram_traffic_increase_pct': None,
                   'stats_source': 'txt' if x_txt else 'runner.log IPC only (.txt missing)'}
            if x_txt:
                row['coverage'] = (base_txt['llc_load_miss'] - x_txt['llc_load_miss']) / base_txt['llc_load_miss']
                row['overprediction'] = (x_txt['llc_load_miss_unique'] + x_txt['llc_pf_miss_unique'] - base_read_miss) / base_read_miss
                row['dram_traffic_increase_pct'] = 100.0 * (x_txt['dram_traffic'] - base_txt['dram_traffic']) / base_txt['dram_traffic']

            x_json = json.load(open(json_path))
            row['l2c_demand_miss_increase'] = x_json['accounting_validation']['l2c_data_demand_misses'] - base_l2_demand_miss
            a = authors.get(stem, {})
            if pref in AUTHORS_EXP and AUTHORS_EXP[pref] in a and 'nopref' in a:
                row['authors_speedup'] = a[AUTHORS_EXP[pref]] / a['nopref']
            rows.append(row)

            if not args.no_json_update:
                diag = x_json['system_efficiency_and_diagnostics']
                diag['net_demand_miss_increase_vs_no_prefetch'] = row['l2c_demand_miss_increase']
                diag['net_demand_miss_increase_definition'] = 'L2C data-demand misses (loads+RFOs), this run minus matching no-prefetch run; negative = fewer misses'
                diag['dram_bandwidth_traffic_increase_pct'] = row['dram_traffic_increase_pct']
                diag['dram_bandwidth_traffic_definition'] = 'DRAM reads+writes (RQ+WQ row-buffer hits+misses), % change vs matching no-prefetch run; null if .txt missing'
                with open(json_path, 'w') as f:
                    json.dump(x_json, f, indent=2)

    fields = ['trace', 'prefetcher', 'ipc', 'ipc_nopref', 'speedup', 'authors_speedup', 'coverage', 'overprediction',
              'dram_traffic_increase_pct', 'l2c_demand_miss_increase', 'stats_source']
    out_csv = os.path.join(args.results_dir, 'summary_vs_nopref.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f'wrote {out_csv} ({len(rows)} rows)')

    print(f"\n{'prefetcher':14} {'geomean speedup':>16} {'authors':>8} {'mean coverage':>14} {'mean overpred':>14}")
    for pref in PREFETCHERS:
        pr = [r for r in rows if r['prefetcher'] == pref]
        if not pr:
            continue
        cov = [r['coverage'] for r in pr if r['coverage'] is not None]
        ovp = [r['overprediction'] for r in pr if r['overprediction'] is not None]
        auth = [r['authors_speedup'] for r in pr if r.get('authors_speedup')]
        print(f"{pref:14} {geomean([r['speedup'] for r in pr]):16.3f} {geomean(auth) if auth else float('nan'):8.3f} "
              f"{sum(cov) / len(cov):14.1%} {sum(ovp) / len(ovp):14.1%}")


if __name__ == '__main__':
    main()
