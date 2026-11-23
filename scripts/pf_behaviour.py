#!/usr/bin/env python3
"""
What each prefetcher issues on one trace, from its own prefetch log (<run_dir>/<stem>_<pref>.pf.bin[.zst]):
requests per trigger access (degree), distinct deltas per trigger (several offsets at once), requested and issued
fill level (L2 vs LLC only), request outcomes and the most common deltas; plus L2C prefetch and DRAM statistics
from the run's .txt (measured region).

Output: <out_dir>/<stem>/pf_behaviour.json

A request is every prefetch-log record except EVICTED_UNUSED. delta = prefetched line - trigger line, as logged.
Bingo queues prefetches in its streamer and issues them on later accesses, so its trigger is the access that
issued a prefetch and its deltas are relative to that access, not to the one that predicted it.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eventlog import NO_TRIGGER, PREFETCH_EVENTS, load_prefetch  # noqa: E402

PREFETCHERS = ['pythia', 'mlop', 'spp_dev', 'bingo', 'next_line', 'ip_stride', 'va_ampm_lite']
EV_EVICTED = PREFETCH_EVENTS.index('EVICTED_UNUSED')
EV_ISSUED_L2 = PREFETCH_EVENTS.index('ISSUED_FILL_L2')
EV_ISSUED_LLC = PREFETCH_EVENTS.index('ISSUED_LLC_ONLY')
TOP_DELTAS = 12


def quantiles(x):
    if len(x) == 0:
        return {'mean': None, 'p50': None, 'p90': None, 'max': None}
    p50, p90 = np.percentile(x, [50, 90]).tolist()
    return {'mean': float(np.mean(x)), 'p50': p50, 'p90': p90, 'max': int(np.max(x))}


def log_behaviour(prefix):
    raw = load_prefetch(prefix)
    is_req = raw['event'] != EV_EVICTED
    out = {'evicted_unused': int((~is_req).sum())}
    ev = raw['event'][is_req]
    out['requests'] = int(len(ev))
    out['outcomes'] = {PREFETCH_EVENTS[e]: int(c) for e, c in zip(*np.unique(ev, return_counts=True))}
    issued_l2, issued_llc = out['outcomes'].get('ISSUED_FILL_L2', 0), out['outcomes'].get('ISSUED_LLC_ONLY', 0)
    out['issued'] = issued_l2 + issued_llc
    out['issued_llc_only_share'] = issued_llc / out['issued'] if out['issued'] else None
    out['requested_llc_only_share'] = float(np.mean(raw['fill_this_level'][is_req] == 0)) if out['requests'] else None
    t = raw['trigger_instr'][is_req]
    d = raw['delta'][is_req]
    del raw, ev
    ok = t != NO_TRIGGER
    out['requests_without_trigger'] = int((~ok).sum())
    t, d = t[ok], d[ok]

    v, c = np.unique(d, return_counts=True)
    top = np.argsort(-c, kind='stable')[:TOP_DELTAS]
    out['top_deltas'] = {int(v[i]): float(c[i] / max(1, len(d))) for i in top}

    # per trigger access: number of requests (degree) and of distinct deltas (offsets used at once)
    order = np.lexsort((d, t))
    t, d = t[order], d[order]
    del order
    new_trig = np.ones(len(t), bool)
    new_trig[1:] = t[1:] != t[:-1]
    new_pair = new_trig.copy()
    new_pair[1:] |= d[1:] != d[:-1]
    starts = np.nonzero(new_trig)[0]
    per_trigger = np.diff(np.append(starts, len(t)))
    distinct = np.add.reduceat(new_pair.astype(np.int64), starts) if len(starts) else np.zeros(0, np.int64)
    out['triggers'] = int(len(starts))
    out['requests_per_trigger'] = quantiles(per_trigger)
    out['distinct_deltas_per_trigger'] = quantiles(distinct)
    out['triggers_with_several_deltas_share'] = float(np.mean(distinct > 1)) if len(distinct) else None
    return out


def txt_stats(path):
    t = open(path).read()
    t = t[t.find('Region of Interest'):]
    m = re.search(r'cpu0->cpu0_L2C PREFETCH REQUESTED:\s+(\d+)\s+ISSUED:\s+(\d+)\s+USEFUL:\s+(\d+)\s+USELESS:\s+(\d+)', t)
    rq = re.search(r'Channel 0 RQ ROW_BUFFER_HIT:\s+(\d+)\s+ROW_BUFFER_MISS:\s+(\d+)', t)
    cyc = re.search(r'CPU 0 cumulative IPC: [\d.]+ instructions: \d+ cycles: (\d+)', t)
    dbus = re.search(r'AVG DBUS CONGESTED CYCLE:\s+([\d.]+)', t)
    hits, misses = int(rq.group(1)), int(rq.group(2))
    return {'l2c_pf_requested': int(m.group(1)), 'l2c_pf_issued': int(m.group(2)), 'l2c_pf_useful': int(m.group(3)), 'l2c_pf_useless': int(m.group(4)),
            'dram_reads': hits + misses, 'dram_read_row_buffer_hit_rate': hits / max(1, hits + misses),
            'dram_reads_per_kcycle': 1000 * (hits + misses) / int(cyc.group(1)), 'dram_avg_dbus_congested_cycle': float(dbus.group(1)) if dbus else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stem')
    ap.add_argument('--run-dir', default='results/pass2_events')
    ap.add_argument('--out-dir', default='results/pass2_events/analysis')
    ap.add_argument('--prefetchers', nargs='+', default=PREFETCHERS)
    a = ap.parse_args()
    res = {'trace': a.stem, 'definitions': __doc__.strip()}
    for p in ['no'] + a.prefetchers:
        prefix = os.path.join(a.run_dir, f'{a.stem}_{p}')
        res[p] = txt_stats(prefix + '.txt')
        if p != 'no':
            res[p].update(log_behaviour(prefix))
    os.makedirs(os.path.join(a.out_dir, a.stem), exist_ok=True)
    with open(os.path.join(a.out_dir, a.stem, 'pf_behaviour.json'), 'w') as f:
        json.dump(res, f, indent=2)
    for p in a.prefetchers:
        r = res[p]
        print(f"{p:13} requests {r['requests']:>11,} per trigger {r['requests_per_trigger']['mean'] or 0:6.2f} distinct deltas "
              f"{r['distinct_deltas_per_trigger']['mean'] or 0:5.2f} llc-only {r['issued_llc_only_share'] or 0:6.1%} top {list(r['top_deltas'])[:4]}")


if __name__ == '__main__':
    main()
