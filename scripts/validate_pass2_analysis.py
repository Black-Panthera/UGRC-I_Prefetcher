#!/usr/bin/env python3
"""Independent consistency checks of pass2_analyze / pass2_report / compare_vs_nopref outputs.
Usage: validate_pass2_analysis.py <run_dir> [analysis_dir]   (e.g. results/pass2_strong results/pass2_strong/analysis_bingo_mlop)"""
import csv, glob, gzip, json, math, os, re, sys
from collections import Counter

RUN = sys.argv[1]
AN = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RUN, 'analysis')
CAUGHT, MISSED = {'1', '2', '4'}, {'5', '6'}
HIT = '0'
CLUSTER_ROWS = 500  # pass2_analyze write_clusters: cl[:500]
problems, notes = [], []


def bad(stem, msg):
    problems.append(f'{stem}: {msg}')


def parse_txt(path):
    t = open(path).read()
    t = t[t.find('Region of Interest Statistics'):]
    ipc = float(re.search(r'CPU 0 cumulative IPC: ([\d.]+)', t).group(1))
    m = {}
    for kind in ('LOAD', 'PREFETCH'):
        g = re.search(rf'cpu0->LLC\s+{kind}\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)\s+MISS_MERGE:\s+(\d+)', t)
        m[kind] = (int(g.group(1)), int(g.group(2)))
    dram = 0
    for q in ('RQ', 'WQ'):
        g = re.search(rf'Channel 0 {q} ROW_BUFFER_HIT:\s+(\d+)\s+ROW_BUFFER_MISS:\s+(\d+)', t)
        dram += int(g.group(1)) + int(g.group(2))
    return ipc, m, dram


def close(a, b, rel=1e-6, abs_=1e-9):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


stems = sorted(os.path.basename(f)[:-len('_no.txt')] for f in glob.glob(os.path.join(RUN, '*_no.txt')))
print(f'{len(stems)} traces')
sets = {tuple(json.load(open(os.path.join(AN, s, 'summary.json'))).get('rule_based', ['spp_dev', 'next_line', 'ip_stride', 'va_ampm_lite'])) for s in stems}
assert len(sets) == 1, sets
RB = list(sets.pop())
PREFS = ['pythia'] + RB
print('rule-based:', RB)
txt = {(s, p): parse_txt(os.path.join(RUN, f'{s}_{p}.txt')) for s in stems for p in ['no'] + PREFS}
nan_delta_tot = Counter()
ooo = Counter()

for s in stems:
    d = os.path.join(AN, s)
    # --- log
    log = open(os.path.join(AN, 'logs', f'{s}.log')).read()
    if 'Exit status: 0' not in log: bad(s, 'nonzero exit')
    if re.search(r'Traceback|Error|nan|inf\b', log.split('Command being timed')[0], re.I):
        notes.append(f'{s}: log mentions error/nan/warning: ' + ' | '.join(l for l in log.splitlines() if re.search(r'Traceback|Error|warn|nan', l, re.I))[:300])
    # --- summary.json
    j = json.load(open(os.path.join(d, 'summary.json')))
    if j['trace'] != s: bad(s, 'trace name mismatch')
    lig = s.startswith('ligra_')
    r = j['region']
    if r['warmup_instructions'] != 100_000_000 or r['measured_instructions'] != (150_000_000 if lig else 200_000_000) \
            or r['trace_length'] != (250_000_000 if lig else None) and r['trace_length'] != (250 if lig else None):
        bad(s, f'region {r}')
    for p in ['no'] + PREFS:
        if j['verification_vs_pass1'][p]['ipc_pass2'] != txt[(s, p)][0]: bad(s, f'IPC {p} summary {j["verification_vs_pass1"][p]["ipc_pass2"]} != txt {txt[(s, p)][0]}')
    po = j['population']
    M = po['misses_matched_in_all_runs']
    if po['nopref_misses'] != M + po['misses_not_matched_in_some_run']: bad(s, 'misses != matched + unmatched')
    if not po['nopref_misses'] <= po['nopref_l2_demand_accesses']: bad(s, 'misses > accesses')
    if max(po['unmatched_by_run'].values()) > po['misses_not_matched_in_some_run'] or \
            sum(po['unmatched_by_run'].values()) < po['misses_not_matched_in_some_run']: bad(s, 'unmatched_by_run inconsistent')
    cat = j['categories']
    if sum(c['count'] for c in cat.values()) != M: bad(s, 'categories do not sum to population')
    if not close(sum(c['share_of_population'] for c in cat.values()), 1.0, 1e-9): bad(s, 'shares != 1')
    pp = j['per_prefetcher']
    for p in PREFS:
        x = pp[p]
        if x['caught'] + x['indirect_hit'] + x['missed'] != M: bad(s, f'{p} caught+indirect+missed != population')
        if x['caught_timely'] + x['caught_late'] > x['caught']: bad(s, f'{p} timely+late > caught')
        if not x['lead_after_access'] <= x['lead_known'] <= x['caught']: bad(s, f'{p} lead counts')
        q = [x['lead_p25'], x['lead_median'], x['lead_p90']]
        if x['lead_known'] > x['lead_after_access'] and not (0 <= q[0] <= q[1] <= q[2]): bad(s, f'{p} lead quantiles {q}')
    py, rb = cat['py_only']['count'], cat['rb_only']['count']
    if py > pp['pythia']['caught'] or any(py > pp[p]['missed'] for p in RB): bad(s, 'py_only exceeds bounds')
    if rb > pp['pythia']['missed']: bad(s, 'rb_only > pythia missed')
    if cat['both']['count'] > min(pp['pythia']['caught'], max(pp[p]['caught'] for p in RB) * 4): bad(s, 'both exceeds bounds')
    pw = j['pairwise_pythia_vs']
    for p in RB:
        if pw[p]['pythia_caught_this_missed'] < py: bad(s, f'pairwise {p} < py_only')
        if pw[p]['this_caught_pythia_missed'] > rb: bad(s, f'pairwise {p} reverse > rb_only')
        b = pw[p]['both_caught']
        if b > cat['both']['count']: bad(s, f'pairwise {p} both_caught > category both')
        if b + pw[p]['pythia_caught_this_missed'] > pp['pythia']['caught'] or b + pw[p]['this_caught_pythia_missed'] > pp[p]['caught']:
            bad(s, f'pairwise {p} both + one-sided > caught')
        if pw[p]['both_this_triggered_earlier'] + pw[p]['both_pythia_triggered_earlier'] > b: bad(s, f'pairwise {p} earlier counts > both')
        if max(pw[p]['both_late_pythia'], pw[p]['both_late_this']) > b: bad(s, f'pairwise {p} late > both')
    if rb > sum(pw[p]['this_caught_pythia_missed'] for p in RB): bad(s, 'rb_only > sum of pairwise reverse')
    for p in RB:
        if sum(j['why_rule_based_missed_py_only'][p].values()) != py: bad(s, f'why {p} sum != py_only')
    if sum(j['why_pythia_missed_rb_only'].values()) != rb: bad(s, 'why pythia sum != rb_only')
    cl = j['clusters']
    for v in ('', '_tight', '_either_order'):
        if sum(cl[f'py_only_access_count_by_rule_label{v}'].values()) != py: bad(s, f'py labels{v} sum')
        if sum(cl[f'rb_only_access_count_by_rule_label{v}'].values()) != rb: bad(s, f'rb labels{v} sum')
    cs = list(cl['py_only_top_cluster_cumulative_share'].values())
    if cs != sorted(cs) or not all(0 <= x <= 1 + 1e-9 for x in cs): bad(s, 'cumulative share not monotone/in range')
    W = j['window_instructions']
    if not (0 < W < 300_000_000): bad(s, f'window {W}')

    # --- per-access CSVs (full decompress = integrity check; recount everything from rows)
    lo, hi = r['warmup_instructions'], r['warmup_instructions'] + r['measured_instructions']
    for name, n, want in (('py_only', py, 'py'), ('rb_only', rb, 'rb')):
        rows, why, nan_d, nan_d_warm, keys, bad_rows, deltas = 0, {p: Counter() for p in PREFS}, 0, 0, set(), 0, Counter()
        with gzip.open(os.path.join(d, f'{name}.csv.gz'), 'rt') as f:
            rd = csv.reader(f)
            hdr = next(rd)
            ix = {h: i for i, h in enumerate(hdr)}
            for row in rd:
                rows += 1
                if len(row) != len(hdr): bad_rows += 1; continue
                iid = int(row[ix['instr_id']])
                if not lo <= iid < hi: bad_rows += 1
                k = (iid, row[ix['pline']])
                if k in keys: bad_rows += 1
                keys.add(k)
                if int(row[ix['pline']], 16) >= 1 << 28: bad_rows += 1
                o = {p: row[ix[f'{p}_outcome']] for p in PREFS}
                if want == 'py':
                    if row[ix['caught_by']] != 'pythia' or o['pythia'] not in CAUGHT or any(o[p] not in MISSED for p in RB): bad_rows += 1
                else:
                    if o['pythia'] not in MISSED or not any(o[p] in CAUGHT for p in RB) or o[row[ix['caught_by']]] not in CAUGHT \
                            or row[ix['caught_by']] != next(p for p in RB if o[p] in CAUGHT): bad_rows += 1
                for p in PREFS: why[p][row[ix[f'{p}_why']]] += 1
                tr = row[ix['trigger_instr']]
                if tr and int(tr) > iid: ooo[name] += 1          # trigger after the access: out-of-order execution, legitimate
                if row[ix['delta']] == '':
                    nan_d += 1
                    if tr and int(tr) < lo: nan_d_warm += 1
                else:
                    deltas[int(row[ix['delta']])] += 1
                if int(row[ix['nopref_latency']]) <= 0: bad_rows += 1
        if rows != n: bad(s, f'{name}.csv rows {rows} != summary {n}')
        if bad_rows: bad(s, f'{name}.csv {bad_rows} inconsistent rows')
        if want == 'py':
            for p in RB:
                if dict(why[p]) != j['why_rule_based_missed_py_only'][p]: bad(s, f'why {p} recount differs')
            top = dict(Counter({(None if k is None else k): v for k, v in deltas.items()}).most_common(12))
            sj = {(None if k in ('None', 'null') else int(k)): v for k, v in j['clusters']['py_only_by_delta'].items()}
            for k, v in top.items():
                if k in sj and sj[k] != v: bad(s, f'delta {k} count {v} != summary {sj[k]}')
        else:
            if dict(why['pythia']) != j['why_pythia_missed_rb_only']: bad(s, 'why pythia recount differs')
        nan_delta_tot[name] += nan_d
        nan_delta_tot[name + '_rows'] += rows
        if n and nan_d / n > 0.02:
            notes.append(f'{s}: {name} delta unknown for {nan_d}/{n} ({100 * nan_d / n:.1f}%), trigger in warmup for {nan_d_warm}')

    # --- cluster CSVs
    for name, n in (('py_only', py), ('rb_only', rb)):
        with open(os.path.join(d, f'clusters_{name}.csv')) as f:
            cr = list(csv.DictReader(f))
        total = j['clusters'][f'{name}_clusters']
        capped = total > CLUSTER_ROWS        # pass2_analyze writes only the top CLUSTER_ROWS clusters
        csum = sum(int(c['count']) for c in cr)
        if (csum > n) if capped else (csum != n): bad(s, f'clusters_{name} counts sum {csum} vs {n}')
        if [int(c['rank']) for c in cr] != list(range(1, len(cr) + 1)): bad(s, f'clusters_{name} ranks')
        if cr and not (float(cr[-1]['cumulative_share']) <= 1 + 1e-6 if capped else close(float(cr[-1]['cumulative_share']), 1.0, 1e-6)):
            bad(s, f'clusters_{name} cumulative end {cr[-1]["cumulative_share"]}')
        if capped and not close(float(cr[-1]['cumulative_share']), csum / n, 1e-6): bad(s, f'clusters_{name} cumulative end != rows / total')
        if sorted((int(c['count']) for c in cr), reverse=True) != [int(c['count']) for c in cr]: bad(s, f'clusters_{name} not sorted')
        if len(cr) != min(total, CLUSTER_ROWS): bad(s, f'clusters_{name} length {len(cr)} vs summary {total}')
        if capped:
            notes.append(f'{s}: clusters_{name} has {total} clusters, CSV keeps the top {CLUSTER_ROWS} ({100 * csum / n:.1f}% of accesses)')
        for c in cr:
            for f_, lab in (('precision_l2', 'rule_label'), ('precision_l2_tight', 'rule_label_tight'), ('precision_l2_either_order', 'rule_label_either_order')):
                if c[f_] in ('', 'None'):
                    continue
                x = float(c[f_])
                if not 0 <= x <= 1: bad(s, f'clusters_{name} rank {c["rank"]} {f_}={x}')
                exp = 'repeatable' if x >= 0.5 else 'partial' if x >= 0.2 else 'lucky'
                if c[lab] != exp: bad(s, f'clusters_{name} rank {c["rank"]} {lab}={c[lab]} but {f_}={x}')
            if c['precision_program'] not in ('', 'None') and not 0 <= float(c['precision_program']) <= 1: bad(s, f'precision_program out of range')
    # --- pattern cards
    pc = open(os.path.join(d, 'pattern_cards.md')).read()
    if not pc.startswith(f'# Pattern cards: {s}') or pc.count('\n### #') < 1: bad(s, 'pattern_cards malformed')
    if f'({py:,} accesses)' not in pc: bad(s, 'pattern_cards py_only count mismatch')

# --- report
rep = open(os.path.join(AN, 'REPORT.md')).read()
for s in stems:
    if s not in rep: problems.append(f'REPORT: {s} missing')
for w in ('nan', 'None', 'inf', 'Traceback'):
    c = len(re.findall(rf'\b{w}\b', rep))
    if c: notes.append(f'REPORT.md contains "{w}" x{c}')

# --- summary_vs_nopref.csv, recomputed from the .txt files
rows = list(csv.DictReader(open(os.path.join(RUN, 'summary_vs_nopref.csv'))))
if len(rows) != len(stems) * len(PREFS): problems.append(f'summary_vs_nopref rows {len(rows)}')
for r_ in rows:
    s, p = r_['trace'], r_['prefetcher']
    ipc, m, dr = txt[(s, p)]
    ipc0, m0, dr0 = txt[(s, 'no')]
    exp = {'ipc': ipc, 'ipc_nopref': ipc0, 'speedup': ipc / ipc0,
           'coverage': (m0['LOAD'][0] - m['LOAD'][0]) / m0['LOAD'][0],
           'overprediction': ((m['LOAD'][0] - m['LOAD'][1] + m['PREFETCH'][0] - m['PREFETCH'][1]) - (m0['LOAD'][0] - m0['LOAD'][1] + m0['PREFETCH'][0] - m0['PREFETCH'][1]))
                             / (m0['LOAD'][0] - m0['LOAD'][1] + m0['PREFETCH'][0] - m0['PREFETCH'][1]),
           'dram_traffic_increase_pct': 100 * (dr - dr0) / dr0}
    for k, v in exp.items():
        if not close(float(r_[k]), v, 1e-9): problems.append(f'vs_nopref {s} {p} {k} {r_[k]} != {v}')
    if r_['stats_source'] != 'txt': problems.append(f'vs_nopref {s} {p} source {r_["stats_source"]}')
    jj = json.load(open(os.path.join(RUN, f'{s}_{p}.json')))
    dg = jj['system_efficiency_and_diagnostics']
    j0 = json.load(open(os.path.join(RUN, f'{s}_no.json')))
    l2d = jj['accounting_validation']['l2c_data_demand_misses'] - j0['accounting_validation']['l2c_data_demand_misses']
    if dg.get('net_demand_miss_increase_vs_no_prefetch') != l2d or int(r_['l2c_demand_miss_increase']) != l2d: problems.append(f'json {s} {p} demand miss field')
    if not close(dg.get('dram_bandwidth_traffic_increase_pct') or 0, exp['dram_traffic_increase_pct'], 1e-9): problems.append(f'json {s} {p} dram field')

print(f'trigger after access (out-of-order): py_only {ooo["py_only"]}, rb_only {ooo["rb_only"]}')
print(f'delta unknown: py_only {nan_delta_tot["py_only"]}/{nan_delta_tot["py_only_rows"]}, rb_only {nan_delta_tot["rb_only"]}/{nan_delta_tot["rb_only_rows"]}')
print('NOTES:'); [print('  ' + n) for n in notes]
print(f'PROBLEMS: {len(problems)}'); [print('  ' + p) for p in problems[:200]]
