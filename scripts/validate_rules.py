#!/usr/bin/env python3
"""
Checks prefetch rules (for example proposed from pattern cards) against one trace's no-prefetch L2 stream: how often each
rule's prefetches would be used, and how many of the residual Pythia-only misses (and of all misses) it would cover.

Rule file (JSON): {"rules": [{"id": "...", "trace": "<stem>", "trigger_pc": "0x...", "offsets": [3, 1],
                              "target_pc": "0x..." (optional), "note": "..."}]}
A rule means: when an access of trigger_pc is passed to the L2 prefetcher at physical line X, prefetch X+offset for every
offset that stays in X's 4 KB page (L2 prefetchers see physical addresses and cannot cross a page).

Per rule, over the no-prefetch run (window = the trace's analysis window from summary.json):
  issued          trigger accesses x in-page offsets
  precision       share of issued lines that any instruction accesses within the window after the trigger
  miss_precision  share of issued lines whose first access within the window was a demand miss without prefetching
  residual_cov    residual accesses (<analysis_dir>/<stem>/py_only.csv.gz: Pythia caught them, every rule-based prefetcher
                  missed them) with a trigger access at line - offset in the window before them
  miss_cov        the same over all no-prefetch L2 demand misses
  program_precision  (rules with target_pc) the same rule over the program's memory stream from the trace (virtual lines,
                  target_pc only), as pass2_analyze's precision_program
  strict window   the same three measures with a per-rule window of 4 x the median trigger->use distance of the residual
                  accesses it covers (min 256): the analysis window is the 99th percentile of Pythia's lead and can be
                  ~0.5M instructions, far longer than a prefetched line survives in the L2, so it inflates both measures
  keep            strict precision >= --min-precision and strict residual_cov >= max(--min-covered, --min-covered-share x
                  residual set)
Per trace: union coverage (strict window) of the kept rules, the upper bound if all were built, and the table they need.
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
from pass2_analyze import MISSED, TraceIndex, load_run, resolve_trace, rule_precision  # noqa: E402

PAGE_LINES = 64
SHIFT = np.int64(1 << 31)


def packed(lines, ids):
    """Sortable (line, instr_id) key; lines < 2^28 and instr_ids < 2^31."""
    return lines.astype(np.int64) * SHIFT + ids.astype(np.int64)


def first_use(acc_key, acc_line, acc_id, acc_miss, lines, after, window):
    """For each (line, trigger instr): is there an access to line within (after, after + window], and was the first one a miss."""
    pos = np.searchsorted(acc_key, packed(lines, after), 'right')
    ok = pos < len(acc_key)
    pos = np.where(ok, pos, 0)
    used = ok & (acc_line[pos] == lines) & (acc_id[pos] - after <= window)
    return used, used & acc_miss[pos]


def covered(trig_key, trig_line, trig_id, lines, ids, offsets, window):
    """For each access (line, instr): a trigger at line - offset (same page) in [instr - window, instr) for some offset,
    and the distance to the nearest such trigger."""
    out = np.zeros(len(lines), bool)
    dist = np.full(len(lines), np.iinfo(np.int64).max)
    for off in offsets:
        want = lines - off
        same_page = (want // PAGE_LINES) == (lines // PAGE_LINES)
        pos = np.searchsorted(trig_key, packed(want, ids), 'left') - 1
        ok = pos >= 0
        pos = np.where(ok, pos, 0)
        hit = same_page & ok & (trig_line[pos] == want) & (ids - trig_id[pos] <= window)
        out |= hit
        dist = np.where(hit, np.minimum(dist, ids - trig_id[pos]), dist)
    return out, dist


def check_trace(stem, rules, a):
    summary = json.load(open(os.path.join(a.analysis_dir, stem, 'summary.json')))
    window = summary['window_instructions']
    base = load_run(a.run_dir, stem, 'no')['acc']
    ids = base['instr_id'].astype(np.int64)
    pcs = {int(r['trigger_pc'], 16) for r in rules} | {int(r['target_pc'], 16) for r in rules if r.get('target_pc')}
    order = np.argsort(ids, kind='stable')
    tix = TraceIndex(resolve_trace(a.trace_dir, stem), int(ids.max()))
    id_ip, _, program, _ = tix.scan(ids=base['instr_id'][order].astype(np.uint64), pcs=pcs, region_start=int(ids.min()))
    pc = np.empty(len(base), np.uint64)
    pc[order] = id_ip

    line = base['pline'].astype(np.int64)
    miss = np.isin(base['outcome'], MISSED)
    acc_order = np.argsort(packed(line, ids), kind='stable')
    acc_key, acc_line, acc_id, acc_miss = packed(line, ids)[acc_order], line[acc_order], ids[acc_order], miss[acc_order]

    res_ids, res_lines = [], []
    with gzip.open(os.path.join(a.analysis_dir, stem, 'py_only.csv.gz'), 'rt') as f:
        for row in csv.DictReader(f):
            res_ids.append(int(row['instr_id']))
            res_lines.append(int(row['pline'], 16))
    res_ids, res_lines = np.array(res_ids, np.int64), np.array(res_lines, np.int64)
    min_cov = max(a.min_covered, a.min_covered_share * len(res_ids))

    rows, union_res, union_miss = [], np.zeros(len(res_ids), bool), np.zeros(int(miss.sum()), bool)
    for r in rules:
        tpc = int(r['trigger_pc'], 16)
        t = (base['seen'] == 1) & (pc == np.uint64(tpc))
        t_line, t_id = line[t], ids[t]
        k = np.argsort(packed(t_line, t_id), kind='stable')
        t_key, t_line_s, t_id_s = packed(t_line, t_id)[k], t_line[k], t_id[k]
        def use(w):
            issued = used = used_miss = 0
            for off in r['offsets']:
                y = t_line + off
                inside = (y // PAGE_LINES) == (t_line // PAGE_LINES)
                u, um = first_use(acc_key, acc_line, acc_id, acc_miss, y[inside], t_id[inside], w)
                issued, used, used_miss = issued + int(inside.sum()), used + int(u.sum()), used_miss + int(um.sum())
            return issued, used, used_miss
        issued, used, used_miss = use(window)
        cov_res, d_res = covered(t_key, t_line_s, t_id_s, res_lines, res_ids, r['offsets'], window)
        cov_miss, _ = covered(t_key, t_line_s, t_id_s, line[miss], ids[miss], r['offsets'], window)
        strict = int(min(window, max(256, 4 * np.median(d_res[cov_res])))) if cov_res.any() else 256
        _, used_s, used_miss_s = use(strict)
        cov_res_s, _ = covered(t_key, t_line_s, t_id_s, res_lines, res_ids, r['offsets'], strict)
        cov_miss_s, _ = covered(t_key, t_line_s, t_id_s, line[miss], ids[miss], r['offsets'], strict)
        prog = None
        if r.get('target_pc'):
            t_ids_p, t_lines_p, first = program[tpc]
            a_ids_p, a_lines_p, _ = program[int(r['target_pc'], 16)]
            p = [rule_precision(t_ids_p[first].astype(np.uint64), t_lines_p[first], a_ids_p.astype(np.uint64), a_lines_p, off, window)[0]
                 for off in r['offsets']]
            prog = float(np.mean([x for x in p if x is not None])) if any(x is not None for x in p) else None
        precision = used / issued if issued else None
        precision_s = used_s / issued if issued else None
        keep = precision_s is not None and precision_s >= a.min_precision and cov_res_s.sum() >= min_cov
        if keep:
            union_res |= cov_res_s
            union_miss |= cov_miss_s
        rows.append({'trace': stem, 'id': r['id'], 'trigger_pc': r['trigger_pc'], 'offsets': r['offsets'], 'target_pc': r.get('target_pc'),
                     'trigger_accesses': int(t.sum()), 'issued': issued, 'precision': precision,
                     'miss_precision': used_miss / issued if issued else None,
                     'residual_covered': int(cov_res.sum()), 'residual_share': float(cov_res.mean()) if len(cov_res) else None,
                     'miss_covered': int(cov_miss.sum()), 'miss_share': float(cov_miss.mean()) if len(cov_miss) else None,
                     'strict_window': strict, 'strict_precision': precision_s, 'strict_miss_precision': used_miss_s / issued if issued else None,
                     'strict_residual_covered': int(cov_res_s.sum()), 'strict_residual_share': float(cov_res_s.mean()) if len(cov_res_s) else None,
                     'strict_miss_covered': int(cov_miss_s.sum()), 'strict_miss_share': float(cov_miss_s.mean()) if len(cov_miss_s) else None,
                     'program_precision': prog, 'keep': bool(keep), 'note': r.get('note', '')})
    kept = [x for x in rows if x['keep']]
    trace_sum = {'trace': stem, 'window_instructions': window, 'residual_accesses': int(len(res_ids)), 'nopref_misses': int(miss.sum()),
                 'rules': len(rules), 'kept': len(kept),
                 'kept_union_residual_covered': int(union_res.sum()), 'kept_union_residual_share': float(union_res.mean()) if len(res_ids) else None,
                 'kept_union_miss_covered': int(union_miss.sum()), 'kept_union_miss_share': float(union_miss.mean()) if len(union_miss) else None,
                 'table_trigger_pcs': len({x['trigger_pc'] for x in kept}), 'table_offsets': sum(len(x['offsets']) for x in kept)}
    return rows, trace_sum


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rules')
    ap.add_argument('--run-dir', default='results/pass2_events')
    ap.add_argument('--analysis-dir', default='results/pass2_events/analysis_bingo_mlop')
    ap.add_argument('--trace-dir', default='test_traces')
    ap.add_argument('--out', required=True, help='output prefix: <out>.csv (per rule) and <out>.json (per trace)')
    ap.add_argument('--min-precision', type=float, default=0.5)
    ap.add_argument('--min-covered', type=int, default=100)
    ap.add_argument('--min-covered-share', type=float, default=0.005)
    a = ap.parse_args()
    assert os.path.abspath(a.out + '.json') != os.path.abspath(a.rules), '--out would overwrite the rule file'
    by_trace = defaultdict(list)
    for r in json.load(open(a.rules))['rules']:
        by_trace[r['trace']].append(r)
    rows, traces = [], []
    for stem, rules in sorted(by_trace.items()):
        if not os.path.exists(os.path.join(a.analysis_dir, stem, 'summary.json')):
            print(f'skip {stem}: no analysis in {a.analysis_dir}')
            continue
        r, t = check_trace(stem, rules, a)
        rows += r
        traces.append(t)
        print(json.dumps(t))
    cols = list(rows[0]) if rows else []
    with open(a.out + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()})
    with open(a.out + '.json', 'w') as f:
        json.dump({'definitions': __doc__.strip(), 'traces': traces}, f, indent=2)
    print(f'wrote {a.out}.csv ({len(rows)} rules), {a.out}.json ({len(traces)} traces)')


if __name__ == '__main__':
    main()
