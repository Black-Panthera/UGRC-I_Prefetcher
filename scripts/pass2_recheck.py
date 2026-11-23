#!/usr/bin/env python3
"""
Re-check an earlier Pythia-only set against prefetcher runs it did not include: for every access in
<old_analysis>/<stem>/py_only.csv.gz (Pythia caught it, the earlier rule-based set missed it), what each of the
given prefetchers did on that same access (instr_id, physical line), overall and per earlier Pythia-only cluster
(access PC, trigger PC, delta): caught at the L2, and cycles saved against the no-prefetch latency (as pass2_analyze:
latency of a miss or late catch is the logged fill time, a hit costs 0), which also counts misses served faster, e.g. by
lines the prefetcher filled into the LLC only.

Output: <out_dir>/<stem>/old_py_only_recheck.json
"""

import argparse
import csv
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pass2_analyze import CAUGHT, HIT, LINE_BITS, MISS, MISS_MERGED, MISSED, PF_LATE, PF_TIMELY, load_run, lookup  # noqa: E402

LEGACY_RULE_BASED = ['spp_dev', 'next_line', 'ip_stride', 'va_ampm_lite']
TOP_CLUSTERS = 20


def outcome_counts(o, lat, lat0):
    m = np.isin(o, MISSED)
    return {'caught': int(np.isin(o, CAUGHT).sum()), 'caught_timely': int((o == PF_TIMELY).sum()), 'caught_late': int((o == PF_LATE).sum()),
            'missed': int(m.sum()), 'hit': int((o == HIT).sum()), 'unmatched': int((o < 0).sum()),
            'missed_at_half_latency_or_less': int((m & (lat <= lat0 // 2)).sum()),
            'cycles_saved_share': float(np.sum((lat0 - lat)[o >= 0]) / max(1, np.sum(lat0[o >= 0])))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stem')
    ap.add_argument('--run-dir', default='results/pass2_events')
    ap.add_argument('--old-analysis', default='results/pass2_events/analysis')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--prefetchers', nargs='+', default=['pythia', 'mlop', 'bingo'], help='pythia gives the reference (it caught all of them)')
    a = ap.parse_args()

    old = os.path.join(a.old_analysis, a.stem)
    old_rb = json.load(open(os.path.join(old, 'summary.json'))).get('rule_based', LEGACY_RULE_BASED)
    instr, pline, cluster, lat0 = [], [], [], []
    with gzip.open(os.path.join(old, 'py_only.csv.gz'), 'rt') as f:
        for row in csv.DictReader(f):
            instr.append(int(row['instr_id']))
            pline.append(int(row['pline'], 16))
            lat0.append(max(0, int(row['nopref_latency'])))
            cluster.append((row['pc'], row['trigger_pc'], int(row['delta']) if row['delta'] else None))
    keys = (np.array(instr, dtype=np.uint64) << np.uint64(LINE_BITS)) | np.array(pline, dtype=np.uint64)
    lat0 = np.array(lat0, dtype=np.int64)

    outc, lat = {}, {}
    for p in a.prefetchers:
        run = load_run(a.run_dir, a.stem, p)
        idx = lookup(run, keys)
        o = np.full(len(keys), -1, np.int8)
        t = np.zeros(len(keys), np.int64)
        o[idx >= 0] = run['acc']['outcome'][idx[idx >= 0]]
        t[idx >= 0] = np.maximum(run['acc']['timing'][idx[idx >= 0]], 0)
        outc[p] = o
        lat[p] = np.where(np.isin(o, (MISS, MISS_MERGED, PF_LATE)), t, 0)
        del run, idx
    any_caught = np.zeros(len(keys), bool)
    for p in a.prefetchers:
        if p != 'pythia':
            any_caught |= np.isin(outc[p], CAUGHT)

    members = defaultdict(list)
    for i, c in enumerate(cluster):
        members[c].append(i)
    clusters = []
    for (pc, tpc, d), ks in sorted(members.items(), key=lambda kv: -len(kv[1]))[:TOP_CLUSTERS]:
        ks = np.array(ks)
        clusters.append({'access_pc': pc, 'trigger_pc': tpc, 'delta': d, 'count': len(ks), 'share': len(ks) / len(keys),
                         **{f'{p}_caught': int(np.isin(outc[p][ks], CAUGHT).sum()) for p in a.prefetchers},
                         **{f'{p}_cycles_saved_share': float(np.sum((lat0 - lat[p])[ks][outc[p][ks] >= 0]) / max(1, np.sum(lat0[ks][outc[p][ks] >= 0])))
                            for p in a.prefetchers},
                         'any_caught': int(any_caught[ks].sum())})

    res = {'trace': a.stem, 'old_rule_based': old_rb, 'checked_prefetchers': a.prefetchers, 'old_py_only': int(len(keys)),
           'by_prefetcher': {p: outcome_counts(outc[p], lat[p], lat0) for p in a.prefetchers}, 'any_caught': int(any_caught.sum()),
           'top_old_clusters': clusters}
    os.makedirs(os.path.join(a.out_dir, a.stem), exist_ok=True)
    with open(os.path.join(a.out_dir, a.stem, 'old_py_only_recheck.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ('old_py_only', 'by_prefetcher', 'any_caught')}, indent=1))


if __name__ == '__main__':
    main()
