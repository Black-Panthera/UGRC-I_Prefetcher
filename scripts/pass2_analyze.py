#!/usr/bin/env python3
"""
Pass-2 analysis of one trace: which L2 demand misses Pythia removes that the rule-based
prefetchers do not (and the reverse), why, and which trigger->target relationships explain them.

Inputs (from run_pass2_events.py): <run_dir>/<stem>_<pref>.{acc,pf}.bin[.zst], .txt
Outputs: <out_dir>/<stem>/ summary.json, py_only.csv.gz, rb_only.csv.gz, clusters_py_only.csv,
         clusters_rb_only.csv, pattern_cards.md

Definitions (fixed before looking at the full-length data; see DEFINITIONS below).
"""

import argparse
import csv
import gzip
import json
import lzma
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eventlog import NO_TRIGGER, load_access, load_prefetch  # noqa: E402

# Order matters only for rb_only attribution (caught_by = first of these that caught the access): strongest first.
# The first analyses (before Bingo and MLOP were ported) used --rule-based spp_dev next_line ip_stride va_ampm_lite.
RULE_BASED = ['mlop', 'spp_dev', 'bingo', 'next_line', 'ip_stride', 'va_ampm_lite']

# AccessOutcome / PrefetchEvent codes (inc/champsim_profiler.h)
HIT, PF_TIMELY, PF_REUSE, UPSTREAM, PF_LATE, MISS, MISS_MERGED = range(7)
EV_REJECTED, EV_FILTERED, EV_MERGED_PF, EV_MERGED_DEMAND, EV_DROPPED, EV_ISSUED_L2, EV_ISSUED_LLC, EV_EVICTED = range(8)
LOAD, RFO = 0, 1
CAUGHT = (PF_TIMELY, PF_REUSE, PF_LATE)
MISSED = (MISS, MISS_MERGED)

LINE_BITS = 28  # physical memory is 16 GiB -> < 2^28 cache lines
REPEATABLE, PARTIAL = 0.5, 0.2
TOP_CLUSTERS = 40
EXAMPLES_PER_CLUSTER = 3
HISTORY = 16
SLICE_RADIUS = 256

DEFINITIONS = {
    'population': 'L2 demand accesses (loads + stores from L1D) that MISS in the no-prefetch run, matched by (instr_id, physical line) in every prefetcher run',
    'caught': 'outcome is a prefetch hit (timely first use, reuse of a prefetched line, or late = merged into an in-flight prefetch)',
    'missed': 'outcome is a demand miss',
    'indirect_hit': 'ordinary hit although the no-prefetch run missed (line resident for another reason); counted separately, never as caught',
    'py_only': 'Pythia caught it and every rule-based prefetcher (rule_based) missed it',
    'rb_only': 'Pythia missed it and at least one rule-based prefetcher caught it; caught_by = the first of them in rule_based order',
    'lead': 'supplementary (added with Bingo and MLOP, to compare timeliness): for a caught access, access instr_id minus the instr_id of the access '
            'that triggered the covering prefetch, i.e. how far ahead in the instruction stream it was requested; negative (trigger after the '
            'access, out-of-order) counted as lead_after_access. Bingo queues prefetches in its streamer, so its trigger is the access that issued '
            'the prefetch, which can be later than the one that predicted it',
    'cycles_saved': 'no-prefetch miss latency minus the run\'s latency (miss/late: logged fill time; hit: 0)',
    'cluster': '(target access PC, trigger PC, delta in cache lines) of the prefetch that caught the access',
    'window': '99th percentile of (access instr_id - trigger instr_id) over all of Pythia\'s caught accesses on this trace',
    'precision_l2': 'over the no-prefetch L2 load stream: fraction of trigger-PC accesses at line X followed by the target PC accessing line X+delta within the window',
    'precision_program': 'same rule over the full program memory stream of the measured region from the trace (virtual lines), i.e. including L1 hits',
    'rule_label': f'repeatable if precision_l2 >= {REPEATABLE}, partial if >= {PARTIAL}, else lucky',
    'precision_l2_tight': 'supplementary (added after seeing full-length windows of ~0.5M instructions): same rule with a per-cluster window of '
                          '4 x the cluster median trigger->use distance (min 256), so chance co-occurrence over a long window is excluded',
    'rule_label_tight': 'same thresholds applied to precision_l2_tight',
    'precision_l2_either_order': 'supplementary (added after mcf showed mirror-image clusters with negative trigger->use distance): strict window, '
                                 'but the target access may come before or after the trigger, since out-of-order execution can send the later '
                                 'instruction of a pair to the L2 first',
    'why_missed': 'for each prefetcher, its own prefetch requests for that line with a trigger inside the window before the access; '
                  'if none: requested_only_after_access (a request triggered after the access, within the window), '
                  'requested_only_long_before (only requests older than the window), or never_requested',
}


# ----------------------------------------------------------------------------------------------- loading

def keys_of(acc):
    assert acc['pline'].max(initial=0) < (1 << LINE_BITS), 'physical line exceeds key packing'
    return (acc['instr_id'].astype(np.uint64) << np.uint64(LINE_BITS)) | acc['pline'].astype(np.uint64)


COMPACT_PF = np.dtype([('trigger_instr', '<u8'), ('target_pline', '<u4'), ('delta', '<i4'), ('event', 'u1')])


def trace_length(stem):
    """Instructions in the trace if the name says so (Ligra: ..._length_250M), else None (SPEC traces are far longer than the region)."""
    m = re.search(r'length_(\d+)M', stem)
    return int(m.group(1)) * 1_000_000 if m else None


def resolve_trace(trace_dir, stem):
    """<trace_dir>/<stem>.champsimtrace.xz, falling back to test_traces/pythia_strong for the strong-win traces."""
    for d in (trace_dir, os.path.join(trace_dir, 'pythia_strong'), 'test_traces/pythia_strong'):
        path = os.path.join(d, f'{stem}.champsimtrace.xz')
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'trace for {stem} not found under {trace_dir}')


def load_run(run_dir, stem, pref):
    prefix = os.path.join(run_dir, f'{stem}_{pref}')
    acc = load_access(prefix)
    acc = acc[(acc['type'] == LOAD) | (acc['type'] == RFO)]
    # When a run ends at the very end of a short trace, ChampSim fetches a few instructions past it and wraps to the start;
    # their instr_ids no longer index the trace file, so drop them
    limit = trace_length(stem)
    if limit is not None:
        acc = acc[acc['instr_id'] < limit]
    k = keys_of(acc)
    order = np.argsort(k, kind='stable')
    return {'acc': acc[order], 'keys': k[order]}


def load_compact_prefetch(run_dir, stem, pref):
    """Only the prefetch-log fields the analysis uses (about half the memory of the raw records)."""
    raw = load_prefetch(os.path.join(run_dir, f'{stem}_{pref}'))
    pf = np.empty(len(raw), dtype=COMPACT_PF)
    for f in COMPACT_PF.names:
        pf[f] = raw[f]
    return pf


def lookup(run, keys):
    """Index of each key in run (or -1)."""
    idx = np.searchsorted(run['keys'], keys)
    idx[idx >= len(run['keys'])] = 0
    found = run['keys'][idx] == keys
    return np.where(found, idx, -1)


def roi_section(txt_path):
    if not os.path.exists(txt_path):
        return None
    text = open(txt_path).read()
    if 'ChampSim completed all CPUs' not in text:
        return None
    return text[text.find('Region of Interest'):]


def roi_ipc(section):
    m = re.search(r'CPU 0 cumulative IPC: ([\d.]+)', section or '')
    return float(m.group(1)) if m else None


# ----------------------------------------------------------------------------------------------- trace

class TraceIndex:
    """Instruction-indexed views of a ChampSim trace (64-byte records: ip, flags, 2 dest mem, 4 src mem).
    Memory lines are returned as 6 columns: 4 load operands then 2 store operands (0 = unused)."""

    CHUNK = 1 << 20

    def __init__(self, path, last_instr):
        self.path = path
        self.last_instr = int(last_instr)

    def scan(self, ids=(), pcs=(), slices=(), region_start=0):
        """ids: sorted instr_ids -> ip and load lines; pcs: collect every memory operand of every execution of these PCs
        from region_start on (returned per PC as (instr_ids u32, lines u64, is_first_operand)); slices: (start, end) ranges to dump."""
        ids = np.asarray(ids, dtype=np.uint64)
        pcs = np.asarray(sorted(set(pcs)), dtype=np.uint64)
        id_ip = np.zeros(len(ids), dtype=np.uint64)
        id_lines = np.zeros((len(ids), 6), dtype=np.uint64)
        pc_hits = []
        slice_rows = defaultdict(list)
        slices = sorted(slices)
        base = 0
        with lzma.open(self.path, 'rb') as fh:
            while base <= self.last_instr:
                buf = fh.read(64 * self.CHUNK)
                n = len(buf) // 64
                if n == 0:
                    break
                rec = np.frombuffer(buf[:n * 64], dtype='<u8').reshape(n, 8)
                ip = rec[:, 0]
                src = np.concatenate([rec[:, 4:8], rec[:, 2:4]], axis=1) >> np.uint64(6)
                lo, hi = np.searchsorted(ids, base), np.searchsorted(ids, base + n)
                if hi > lo:
                    pos = (ids[lo:hi] - np.uint64(base)).astype(np.int64)
                    id_ip[lo:hi] = ip[pos]
                    id_lines[lo:hi] = src[pos]
                if len(pcs) and base + n > region_start:
                    sel = np.nonzero(np.isin(ip, pcs))[0]
                    sel = sel[sel + base >= region_start]
                    if len(sel):
                        lines = src[sel]
                        nz = lines != 0
                        first = np.zeros_like(nz)
                        first[np.arange(len(sel)), np.argmax(nz, axis=1)] = True
                        first &= nz
                        rows = np.nonzero(nz)
                        pc_hits.append((np.searchsorted(pcs, ip[sel])[rows[0]].astype(np.uint8),
                                        (sel[rows[0]] + base).astype(np.uint32), lines[rows], first[rows]))
                for (s, e) in slices:
                    if e < base or s >= base + n:
                        continue
                    for i in range(max(s, base), min(e, base + n - 1) + 1):
                        r = rec[i - base]
                        slice_rows[(s, e)].append((i, int(r[0]), [int(x) >> 6 for x in r[4:8] if x], [int(x) >> 6 for x in r[2:4] if x]))
                base += n
        assert len(pcs) < 256
        program = {}
        if pc_hits:
            which = np.concatenate([h[0] for h in pc_hits])
            ops = [np.concatenate([h[i] for h in pc_hits]) for i in (1, 2, 3)]
            del pc_hits
            order = np.argsort(which, kind='stable')
            which = which[order]
            ops = [o[order] for o in ops]
            bounds = np.searchsorted(which, np.arange(len(pcs) + 1))
            for j, pc in enumerate(pcs.tolist()):
                program[pc] = tuple(o[bounds[j]:bounds[j + 1]] for o in ops)
        empty = (np.zeros(0, np.uint32), np.zeros(0, np.uint64), np.zeros(0, bool))
        return id_ip, id_lines, defaultdict(lambda: empty, program), slice_rows


# ----------------------------------------------------------------------------------------------- analysis

def why_missed(pf, plines, instrs, window):
    """Per access: what this prefetcher did about the line inside [instr - window, instr]."""
    out = np.full(len(plines), 'never_requested', dtype=object)
    if pf is None or len(pf) == 0:
        return out
    pf = pf[pf['trigger_instr'] != NO_TRIGGER]
    order = np.lexsort((pf['trigger_instr'], pf['target_pline']))
    tl, ti, ev = pf['target_pline'][order], pf['trigger_instr'][order], pf['event'][order]
    lo = np.searchsorted(tl, plines, 'left')
    hi = np.searchsorted(tl, plines, 'right')
    names = {EV_REJECTED: 'requested_dropped_queue_full', EV_ISSUED_LLC: 'issued_llc_only', EV_DROPPED: 'requested_dropped_mshr_full'}
    for k in range(len(plines)):
        if lo[k] == hi[k]:
            continue
        seg_t, seg_e = ti[lo[k]:hi[k]], ev[lo[k]:hi[k]]
        sel = (seg_t <= instrs[k]) & (seg_t + np.uint64(window) >= instrs[k])
        if not sel.any():
            after = (seg_t > instrs[k]) & (seg_t <= instrs[k] + np.uint64(window))
            out[k] = 'requested_only_after_access' if after.any() else 'requested_only_long_before'
            continue
        e = set(seg_e[sel].tolist())
        if EV_EVICTED in e:
            out[k] = 'issued_then_evicted_unused'
        elif EV_MERGED_DEMAND in e:
            out[k] = 'issued_too_late'
        elif EV_ISSUED_L2 in e:
            out[k] = 'issued_but_missed'
        elif EV_ISSUED_LLC in e:
            out[k] = names[EV_ISSUED_LLC]
        elif EV_REJECTED in e:
            out[k] = names[EV_REJECTED]
        elif EV_DROPPED in e:
            out[k] = names[EV_DROPPED]
        else:
            out[k] = 'requested_line_already_present'
    return out


def prefetch_delta(pf, plines, triggers):
    """Delta of the prefetch (target, trigger) from the prefetcher's own log; NaN if not found."""
    out = np.full(len(plines), np.nan)
    if pf is None:
        return out
    iss = pf[(pf['event'] == EV_ISSUED_L2) | (pf['event'] == EV_ISSUED_LLC) | (pf['event'] == EV_FILTERED) |
             (pf['event'] == EV_MERGED_PF) | (pf['event'] == EV_MERGED_DEMAND)]
    k = (iss['target_pline'].astype(np.uint64) << np.uint64(32)) ^ iss['trigger_instr'].astype(np.uint64)
    order = np.argsort(k)
    k, d = k[order], iss['delta'][order]
    q = (plines.astype(np.uint64) << np.uint64(32)) ^ triggers.astype(np.uint64)
    i = np.searchsorted(k, q)
    i[i >= len(k)] = 0
    ok = (len(k) > 0) & (k[i] == q) if len(k) else np.zeros(len(q), bool)
    out[ok] = d[i[ok]]
    return out


def first_line(lines):
    """First non-zero memory line per instruction (loads before stores)."""
    if len(lines) == 0:
        return np.zeros(0, dtype=np.uint64)
    j = np.argmax(lines != 0, axis=1)
    return lines[np.arange(len(lines)), j]


def most_common_share(x):
    if len(x) < 2:
        return None, 0.0
    v, c = np.unique(np.diff(x.astype(np.int64)), return_counts=True)
    j = int(np.argmax(c))
    return int(v[j]), float(c[j] / c.sum())


def rule_precision(trig_ids, trig_lines, tgt_ids, tgt_lines, delta, window):
    """Fraction of trigger accesses (id, line) followed by a target access at line+delta within window."""
    if len(trig_ids) == 0:
        return None, 0
    if len(tgt_ids) == 0:
        return 0.0, len(trig_ids)
    want = trig_lines.astype(np.int64) + int(delta)
    ids = trig_ids.astype(np.int64)
    tl, ti = tgt_lines.astype(np.int64), tgt_ids.astype(np.int64)
    order = np.lexsort((ti, tl))
    tl, ti = tl[order], ti[order]
    # first target access strictly after the trigger, to the wanted line: search on the packed (line, id) order
    pos = np.searchsorted(tl * (1 << 31) + ti, want * (1 << 31) + ids, 'right')
    ok = pos < len(tl)
    pos = np.where(ok, pos, 0)
    hit = ok & (tl[pos] == want) & (ti[pos] - ids <= window)
    return float(hit.mean()), len(want)


def rule_precision_either_order(trig_ids, trig_lines, tgt_ids, tgt_lines, delta, window):
    """Like rule_precision, but the target access may come before or after the trigger (within +-window):
    out-of-order execution can send the later instruction of a pair to the L2 first."""
    if len(trig_ids) == 0:
        return None
    if len(tgt_ids) == 0:
        return 0.0
    want = trig_lines.astype(np.int64) + int(delta)
    ids = trig_ids.astype(np.int64)
    tl, ti = tgt_lines.astype(np.int64), tgt_ids.astype(np.int64)
    order = np.lexsort((ti, tl))
    tl, ti = tl[order], ti[order]
    packed = tl * (1 << 31) + ti
    pos = np.searchsorted(packed, want * (1 << 31) + (ids - window), 'left')
    ok = pos < len(tl)
    pos = np.where(ok, pos, 0)
    hit = ok & (tl[pos] == want) & (ti[pos] <= ids + window) & ~((ti[pos] == ids) & (int(delta) == 0))
    return float(hit.mean())


def analyze(stem, run_dir, out_dir, trace_dir, pass1_dir, rule_based=RULE_BASED):
    all_prefs = ['pythia'] + list(rule_based)
    os.makedirs(out_dir, exist_ok=True)
    # One run in memory at a time: keep only per-access columns aligned to the no-prefetch run, plus compact prefetch logs
    base_run = load_run(run_dir, stem, 'no')
    base, base_keys = base_run['acc'], base_run['keys']
    del base_run
    runs = {}
    summary = {'trace': stem, 'definitions': DEFINITIONS}
    txt = open(os.path.join(run_dir, f'{stem}_no.txt')).read()
    region = {k: int(m.group(1)) for k, pat in (('warmup_instructions', r'Warmup Instructions: (\d+)'),
                                                 ('measured_instructions', r'Simulation Instructions: (\d+)'))
              if (m := re.search(pat, txt))}
    region['trace_length'] = trace_length(stem)
    summary['region'] = region

    # -- verification against pass 1 (identical simulation => identical ROI statistics)
    verify = {}
    for p in ['no'] + all_prefs:
        new = roi_section(os.path.join(run_dir, f'{stem}_{p}.txt'))
        old = roi_section(os.path.join(pass1_dir, f'{stem}_{p}.txt')) if pass1_dir else None
        strip = lambda s: re.sub(r'CPU 0 runs .*\n', '', s or '')
        verify[p] = {'ipc_pass2': roi_ipc(new), 'ipc_pass1': roi_ipc(old),
                     'roi_stats_identical': (strip(new) == strip(old)) if (new and old) else None}
    summary['verification_vs_pass1'] = verify

    # -- per-run outcomes for every no-prefetch access
    outc, timing, trig = {}, {}, {}
    matched_all = np.ones(len(base), bool)
    for p in all_prefs:
        run = load_run(run_dir, stem, p)
        idx = lookup(run, base_keys)
        m = idx >= 0
        matched_all &= m
        o = np.full(len(base), -1, np.int8)
        t = np.zeros(len(base), np.int32)
        g = np.full(len(base), NO_TRIGGER, np.uint64)
        a = run['acc']
        o[m], t[m], g[m] = a['outcome'][idx[m]], a['timing'][idx[m]], a['trigger_instr'][idx[m]]
        outc[p], timing[p], trig[p] = o, t, g
        del run, a, idx, m
        runs[p] = {'pf': load_compact_prefetch(run_dir, stem, p)}

    base_miss = np.isin(base['outcome'], MISSED)
    pop = base_miss & matched_all
    caught = {p: np.isin(outc[p], CAUGHT) for p in all_prefs}
    missed = {p: np.isin(outc[p], MISSED) for p in all_prefs}
    rb_caught = np.zeros(len(base), bool)
    rb_all_missed = np.ones(len(base), bool)
    for p in rule_based:
        rb_caught |= caught[p]
        rb_all_missed &= missed[p]
    cats = {
        'py_only': pop & caught['pythia'] & rb_all_missed,
        'rb_only': pop & missed['pythia'] & rb_caught,
        'both': pop & caught['pythia'] & rb_caught,
        'none': pop & missed['pythia'] & rb_all_missed,
    }
    cats['other_mixed'] = pop & ~(cats['py_only'] | cats['rb_only'] | cats['both'] | cats['none'])
    L0 = np.maximum(base['timing'].astype(np.int64), 0)

    def saved(p, mask):
        lat = np.where(np.isin(outc[p], (MISS, MISS_MERGED, PF_LATE)), np.maximum(timing[p], 0), 0)
        return int(np.sum((L0 - lat)[mask]))

    summary['population'] = {
        'nopref_l2_demand_accesses': int(len(base)), 'nopref_misses': int(base_miss.sum()),
        'misses_matched_in_all_runs': int(pop.sum()),
        'misses_not_matched_in_some_run': int((base_miss & ~matched_all).sum()),
        'unmatched_by_run': {p: int((base_miss & (outc[p] < 0)).sum()) for p in all_prefs},
    }
    summary['categories'] = {k: {'count': int(v.sum()), 'share_of_population': float(v.sum() / max(1, pop.sum())),
                                 'cycles_saved_by_pythia': saved('pythia', v),
                                 'cycles_saved_by_best_rule_based': max(saved(p, v) for p in rule_based)}
                             for k, v in cats.items()}
    def lead(p, mask):
        """Trigger -> access distance (instructions) of p's prefetches that caught the accesses in mask."""
        m = mask & caught[p] & (trig[p] != NO_TRIGGER)
        d = base['instr_id'][m].astype(np.int64) - trig[p][m].astype(np.int64)
        ahead = d[d >= 0]
        q = np.percentile(ahead, [25, 50, 90]).tolist() if len(ahead) else [None] * 3
        return {'lead_known': int(m.sum()), 'lead_after_access': int((d < 0).sum()), 'lead_p25': q[0], 'lead_median': q[1], 'lead_p90': q[2]}

    def paired(p):
        """On the accesses both Pythia and p caught: whose covering prefetch was triggered earlier, and who was late."""
        both = pop & caught['pythia'] & caught[p]
        k = both & (trig['pythia'] != NO_TRIGGER) & (trig[p] != NO_TRIGGER)
        return {'both_caught': int(both.sum()),
                'both_this_triggered_earlier': int((k & (trig[p] < trig['pythia'])).sum()),
                'both_pythia_triggered_earlier': int((k & (trig['pythia'] < trig[p])).sum()),
                'both_late_pythia': int((both & (outc['pythia'] == PF_LATE)).sum()), 'both_late_this': int((both & (outc[p] == PF_LATE)).sum()),
                'both_lead_median_pythia': lead('pythia', both)['lead_median'], 'both_lead_median_this': lead(p, both)['lead_median']}

    summary['rule_based'] = list(rule_based)
    summary['per_prefetcher'] = {
        p: {'caught': int((pop & caught[p]).sum()), 'caught_timely': int((pop & (outc[p] == PF_TIMELY)).sum()),
            'caught_late': int((pop & (outc[p] == PF_LATE)).sum()), 'indirect_hit': int((pop & (outc[p] == HIT)).sum()),
            'missed': int((pop & missed[p]).sum()), 'cycles_saved_on_population': saved(p, pop),
            'pollution_misses': int((~base_miss & (base['outcome'] == HIT) & missed[p]).sum()), **lead(p, pop)}
        for p in all_prefs}
    summary['pairwise_pythia_vs'] = {
        p: {'pythia_caught_this_missed': int((pop & caught['pythia'] & missed[p]).sum()),
            'this_caught_pythia_missed': int((pop & caught[p] & missed['pythia']).sum()), **paired(p)} for p in rule_based}

    # -- window from Pythia's own caught accesses
    pc_mask = pop & caught['pythia'] & (trig['pythia'] != NO_TRIGGER)
    dist = base['instr_id'][pc_mask].astype(np.int64) - trig['pythia'][pc_mask].astype(np.int64)
    dist = dist[dist >= 0]
    window = int(np.clip(np.percentile(dist, 99), 1000, 1_000_000)) if len(dist) else 10000
    summary['window_instructions'] = window

    # -- trace lookups: PCs of every no-prefetch access and of every trigger we need
    trigger_ids = set()
    for p in all_prefs:
        t = trig[p][pop & caught[p]]
        trigger_ids.update(t[t != NO_TRIGGER].tolist())
    need = np.unique(np.concatenate([base['instr_id'].astype(np.uint64), np.array(sorted(trigger_ids), dtype=np.uint64)]))
    tix = TraceIndex(resolve_trace(trace_dir, stem), need[-1] if len(need) else 0)
    id_ip, _, _, _ = tix.scan(ids=need)
    pc_of = lambda ids: id_ip[np.searchsorted(need, ids.astype(np.uint64))]
    base_pc = pc_of(base['instr_id'])

    def rows_for(mask, catcher):
        """Per-access rows for a category; catcher(p) chooses which run's trigger describes the catch."""
        idx = np.nonzero(mask)[0]
        res = {'idx': idx, 'instr': base['instr_id'][idx].astype(np.uint64), 'pline': base['pline'][idx], 'vline': base['vline'][idx],
               'pc': base_pc[idx], 'lat0': L0[idx]}
        res['catcher'] = np.array([catcher(i) for i in idx], dtype=object)
        trg = np.array([trig[c][i] for c, i in zip(res['catcher'], idx)], dtype=np.uint64)
        res['trigger'] = trg
        res['trigger_pc'] = np.where(trg != NO_TRIGGER, pc_of(np.where(trg != NO_TRIGGER, trg, need[0])), 0)
        res['delta'] = np.full(len(idx), np.nan)
        for c in set(res['catcher'].tolist()):
            s = res['catcher'] == c
            res['delta'][s] = prefetch_delta(runs[c]['pf'], res['pline'][s], trg[s])
        res['distance'] = res['instr'].astype(np.int64) - trg.astype(np.int64)
        res['why'] = {p: why_missed(runs[p]['pf'], res['pline'], res['instr'], window) for p in all_prefs}
        return res

    py = rows_for(cats['py_only'], lambda i: 'pythia')
    first_rb = lambda i: next(p for p in rule_based if caught[p][i])
    rb = rows_for(cats['rb_only'], first_rb)

    def write_rows(path, r):
        with gzip.open(path, 'wt', newline='') as f:
            w = csv.writer(f)
            w.writerow(['instr_id', 'pc', 'pline', 'vline', 'nopref_latency', 'caught_by', 'trigger_instr', 'trigger_pc', 'delta', 'distance_instr']
                       + [f'{p}_outcome' for p in all_prefs] + [f'{p}_why' for p in all_prefs])
            for k, i in enumerate(r['idx']):
                w.writerow([int(r['instr'][k]), hex(int(r['pc'][k])), hex(int(r['pline'][k])), hex(int(r['vline'][k])), int(r['lat0'][k]), r['catcher'][k],
                            int(r['trigger'][k]) if r['trigger'][k] != NO_TRIGGER else '', hex(int(r['trigger_pc'][k])),
                            '' if np.isnan(r['delta'][k]) else int(r['delta'][k]), int(r['distance'][k]) if r['trigger'][k] != NO_TRIGGER else '']
                           + [int(outc[p][i]) for p in all_prefs] + [r['why'][p][k] for p in all_prefs])

    write_rows(os.path.join(out_dir, 'py_only.csv.gz'), py)
    write_rows(os.path.join(out_dir, 'rb_only.csv.gz'), rb)
    summary['why_rule_based_missed_py_only'] = {p: dict(Counter(py['why'][p].tolist()).most_common()) for p in rule_based}
    summary['why_pythia_missed_rb_only'] = dict(Counter(rb['why']['pythia'].tolist()).most_common())

    # -- clusters
    def clusters(r):
        groups = defaultdict(list)
        for k in range(len(r['idx'])):
            d = None if np.isnan(r['delta'][k]) else int(r['delta'][k])
            groups[(int(r['pc'][k]), int(r['trigger_pc'][k]), d, r['catcher'][k])].append(k)
        total = max(1, len(r['idx']))
        out = []
        for (pc, tpc, d, c), ks in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            ks = np.array(ks)
            sv = int(np.sum(r['lat0'][ks]))
            out.append({'access_pc': pc, 'trigger_pc': tpc, 'delta': d, 'caught_by': c, 'count': len(ks), 'share': len(ks) / total,
                        'nopref_cycles': sv, 'same_pc_trigger': pc == tpc,
                        'median_distance': int(np.median(r['distance'][ks])) if len(ks) else None, 'members': ks})
        acc = 0.0
        for c in out:
            acc += c['share']
            c['cumulative_share'] = acc
        return out

    cpy, crb = clusters(py), clusters(rb)

    # -- per-cluster rule checks on the L2 view (no-prefetch run) and the program view (trace)
    top = [c for c in cpy[:TOP_CLUSTERS] if c['delta'] is not None and c['trigger_pc']] + \
          [c for c in crb[:TOP_CLUSTERS // 2] if c['delta'] is not None and c['trigger_pc']]
    pcs = {c['access_pc'] for c in top} | {c['trigger_pc'] for c in top}
    examples = []
    for c in cpy[:TOP_CLUSTERS // 2]:
        for k in c['members'][:EXAMPLES_PER_CLUSTER]:
            examples.append(int(py['instr'][k]))
    for c in crb[:TOP_CLUSTERS // 4]:
        for k in c['members'][:EXAMPLES_PER_CLUSTER]:
            examples.append(int(rb['instr'][k]))
    slices = [(max(0, e - SLICE_RADIUS), e + 8) for e in examples]
    region_start = int(base['instr_id'].min()) if len(base) else 0
    _, _, program, slice_rows = tix.scan(pcs=pcs, slices=slices, region_start=region_start)

    def prog_first(pc):
        ids, lines, first = program[pc]
        return ids[first].astype(np.uint64), lines[first]

    l2_loads = base['seen'] == 1
    for c in top:
        a_pc, t_pc, d = c['access_pc'], c['trigger_pc'], c['delta']
        ts = l2_loads & (base_pc == t_pc)
        tg = (base_pc == a_pc)
        c['precision_l2'], c['trigger_count_l2'] = rule_precision(base['instr_id'][ts], base['pline'][ts], base['instr_id'][tg], base['pline'][tg], d, window)
        c['tight_window'] = int(min(window, max(256, 4 * (c['median_distance'] or 0))))
        c['precision_l2_tight'], _ = rule_precision(base['instr_id'][ts], base['pline'][ts], base['instr_id'][tg], base['pline'][tg], d, c['tight_window'])
        c['precision_l2_either_order'] = rule_precision_either_order(base['instr_id'][ts], base['pline'][ts], base['instr_id'][tg], base['pline'][tg],
                                                                     d, c['tight_window'])
        pe = c['precision_l2_either_order']
        c['rule_label_either_order'] = None if pe is None else ('repeatable' if pe >= REPEATABLE else 'partial' if pe >= PARTIAL else 'lucky')
        pt = c['precision_l2_tight']
        c['rule_label_tight'] = None if pt is None else ('repeatable' if pt >= REPEATABLE else 'partial' if pt >= PARTIAL else 'lucky')
        # program view (measured region): every execution of the trigger PC (first memory operand) vs every memory operand of the target PC
        t_ids, t_lines = prog_first(t_pc)
        a_ids, a_lines, _ = program[a_pc]
        c['precision_program'], c['trigger_count_program'] = rule_precision(t_ids, t_lines, a_ids.astype(np.uint64), a_lines, d, window)
        p = c['precision_l2']
        c['rule_label'] = None if p is None else ('repeatable' if p >= REPEATABLE else 'partial' if p >= PARTIAL else 'lucky')
        # L1 filtering: how much of the target PC's program stream reaches the L2, and how regular it looks in each view
        prog_a = prog_first(a_pc)[1]
        l2_a = base['vline'][tg & l2_loads]
        c['target_pc_l2_fraction'] = float(len(l2_a) / max(1, len(prog_a)))
        c['target_stride_program'], c['target_stride_share_program'] = most_common_share(prog_a)
        c['target_stride_l2'], c['target_stride_share_l2'] = most_common_share(l2_a)

    # -- cross-run summaries per cluster: what the rule-based prefetchers did
    for r, cl in ((py, cpy), (rb, crb)):
        for c in cl[:TOP_CLUSTERS]:
            ks = c['members']
            c['why_missed'] = {p: dict(Counter(r['why'][p][ks].tolist()).most_common(3)) for p in all_prefs if p != c['caught_by']}

    def write_clusters(path, cl):
        cols = ['rank', 'access_pc', 'trigger_pc', 'delta', 'caught_by', 'count', 'share', 'cumulative_share', 'nopref_cycles', 'same_pc_trigger',
                'median_distance', 'precision_l2', 'trigger_count_l2', 'precision_program', 'trigger_count_program', 'rule_label',
                'tight_window', 'precision_l2_tight', 'rule_label_tight', 'precision_l2_either_order', 'rule_label_either_order',
                'target_pc_l2_fraction', 'target_stride_program', 'target_stride_share_program', 'target_stride_l2', 'target_stride_share_l2', 'why_missed']
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(cols)
            for rank, c in enumerate(cl[:500], 1):
                row = dict(c, rank=rank, access_pc=hex(c['access_pc']), trigger_pc=hex(c['trigger_pc']))
                w.writerow([json.dumps(row.get(k)) if k == 'why_missed' else row.get(k) for k in cols])

    write_clusters(os.path.join(out_dir, 'clusters_py_only.csv'), cpy)
    write_clusters(os.path.join(out_dir, 'clusters_rb_only.csv'), crb)

    def coverage_by_label(cl, n_total, key='rule_label'):
        lab = Counter()
        for c in cl[:TOP_CLUSTERS]:
            lab[c.get(key) or 'unchecked'] += c['count']
        lab['beyond_top_clusters'] = n_total - sum(lab.values())
        return dict(lab)

    summary['clusters'] = {
        'py_only_clusters': len(cpy), 'rb_only_clusters': len(crb),
        'py_only_top_cluster_cumulative_share': {n: cpy[min(n, len(cpy)) - 1]['cumulative_share'] for n in (1, 5, 10, 20, 40) if cpy},
        'py_only_access_count_by_rule_label': coverage_by_label(cpy, len(py['idx'])),
        'rb_only_access_count_by_rule_label': coverage_by_label(crb, len(rb['idx'])),
        'py_only_access_count_by_rule_label_tight': coverage_by_label(cpy, len(py['idx']), 'rule_label_tight'),
        'rb_only_access_count_by_rule_label_tight': coverage_by_label(crb, len(rb['idx']), 'rule_label_tight'),
        'py_only_access_count_by_rule_label_either_order': coverage_by_label(cpy, len(py['idx']), 'rule_label_either_order'),
        'rb_only_access_count_by_rule_label_either_order': coverage_by_label(crb, len(rb['idx']), 'rule_label_either_order'),
        'py_only_by_delta': dict(Counter(int(x) if not np.isnan(x) else None for x in py['delta']).most_common(12)),
        'py_only_same_pc_trigger_share': float(np.mean(py['pc'] == py['trigger_pc'])) if len(py['idx']) else None,
    }

    # -- pattern cards (LLM input)
    with open(os.path.join(out_dir, 'pattern_cards.md'), 'w') as f:
        f.write(f'# Pattern cards: {stem}\n\nWindow: {window} instructions. Definitions: see summary.json.\n\n')
        for title, r, cl, n in ((f'Pythia caught, all {len(rule_based)} rule-based missed', py, cpy, TOP_CLUSTERS // 2),
                                ('A rule-based prefetcher caught, Pythia missed', rb, crb, TOP_CLUSTERS // 4)):
            f.write(f'## {title} ({len(r["idx"]):,} accesses)\n\n')
            for rank, c in enumerate(cl[:n], 1):
                f.write(f'### #{rank}: access PC {hex(c["access_pc"])} <- trigger PC {hex(c["trigger_pc"])}, delta {c["delta"]} lines, caught by {c["caught_by"]}\n\n')
                f.write(f'- accesses: {c["count"]:,} ({100 * c["share"]:.1f}% of category, cumulative {100 * c["cumulative_share"]:.1f}%), '
                        f'median trigger->use distance {c["median_distance"]} instructions, same-PC trigger: {c["same_pc_trigger"]}\n')
                if c.get('precision_l2') is not None:
                    f.write(f'- rule "trigger PC at line X -> access PC at X{c["delta"]:+d} within {window} instr": precision {100 * c["precision_l2"]:.1f}% '
                            f'over {c["trigger_count_l2"]:,} L2-visible triggers ({c["rule_label"]}); program view {100 * (c["precision_program"] or 0):.1f}% '
                            f'over {c["trigger_count_program"]:,} executions; strict window {c["tight_window"]} instr: '
                            f'{100 * (c["precision_l2_tight"] or 0):.1f}% ({c["rule_label_tight"]}); either order: '
                            f'{100 * (c["precision_l2_either_order"] or 0):.1f}% ({c["rule_label_either_order"]})\n')
                    f.write(f'- access PC: {100 * c["target_pc_l2_fraction"]:.1f}% of its executions reach the L2; most common stride '
                            f'program {c["target_stride_program"]} ({100 * c["target_stride_share_program"]:.0f}%), '
                            f'L2 view {c["target_stride_l2"]} ({100 * c["target_stride_share_l2"]:.0f}%)\n')
                f.write(f'- other prefetchers: {json.dumps(c.get("why_missed", {}))}\n\n')
                for k in c['members'][:EXAMPLES_PER_CLUSTER]:
                    e = int(r['instr'][k])
                    f.write(f'Example: instr {e}, vline {hex(int(r["vline"][k]))}, pline {hex(int(r["pline"][k]))}, no-prefetch latency {int(r["lat0"][k])} cycles, '
                            f'trigger instr {int(r["trigger"][k]) if r["trigger"][k] != NO_TRIGGER else None}\n\n')
                    sel = (base_pc == c['access_pc']) & (base['instr_id'] <= np.uint64(e))
                    hist = base['vline'][sel][-HISTORY:].astype(np.int64)
                    f.write(f'- last {len(hist)} L2-visible accesses by this PC (vline deltas): {np.diff(hist).tolist()}\n')
                    a_ids_f, a_lines_f = prog_first(c['access_pc'])
                    ph = a_lines_f[a_ids_f <= np.uint64(e)][-HISTORY:].astype(np.int64)
                    f.write(f'- last {len(ph)} program executions of this PC (vline deltas): {np.diff(ph).tolist()}\n')
                    rows = slice_rows.get((max(0, e - SLICE_RADIUS), e + 8), [])
                    gaps = np.diff(a_ids_f.astype(np.int64))
                    loop = int(np.median(gaps)) if len(gaps) else 32
                    loop = int(min(max(loop, 8), SLICE_RADIUS))
                    f.write(f'- code slice, memory instructions only (one loop iteration ~{loop} instr before the access; columns: offset, PC, load vlines, store vlines):\n\n```\n')
                    for (i, ip, lds, sts) in rows:
                        is_mark = i == e or (r['trigger'][k] != NO_TRIGGER and i == int(r['trigger'][k]))
                        if i < e - loop or (not lds and not sts and not is_mark):
                            continue
                        mark = '  <== access' if i == e else ('  <== trigger' if r['trigger'][k] != NO_TRIGGER and i == int(r['trigger'][k]) else '')
                        f.write(f'{i - e:+5d} {hex(ip):>12} ld {",".join(hex(x) for x in lds) or "-":<30} st {",".join(hex(x) for x in sts) or "-"}{mark}\n')
                    f.write('```\n\n')

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stem')
    ap.add_argument('--run-dir', default='results/pass2_events')
    ap.add_argument('--out-dir', default='results/pass2_events/analysis')
    ap.add_argument('--trace-dir', default='test_traces')
    ap.add_argument('--pass1-dir', default='results/pass1')
    ap.add_argument('--rule-based', nargs='+', default=RULE_BASED, help='rule-based prefetcher runs to compare Pythia with, strongest first')
    a = ap.parse_args()
    s = analyze(a.stem, a.run_dir, os.path.join(a.out_dir, a.stem), a.trace_dir, a.pass1_dir, a.rule_based)
    print(json.dumps({k: s[k] for k in ('population', 'categories', 'window_instructions', 'clusters')}, indent=1, default=str))


if __name__ == '__main__':
    main()
