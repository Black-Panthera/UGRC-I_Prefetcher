#!/usr/bin/env python3
"""Cross-trace report for pass 2: reads <analysis_dir>/<stem>/summary.json + clusters_*.csv, writes REPORT.md."""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict



def program_of(stem):
    """Benchmark a trace belongs to: '429.mcf-184B' -> '429.mcf', '620.omnetpp_s-141B' -> '620.omnetpp_s',
    'ligra_PageRankDelta.com-lj...' -> 'ligra_PageRankDelta'."""
    return stem.split('.com-')[0] if stem.startswith('ligra_') else stem.rsplit('-', 1)[0]


def input_pairs(stems):
    """Every pair of traces from the same program (different inputs or phases), for the cross-input check."""
    groups = defaultdict(list)
    for s in sorted(stems):
        groups[program_of(s)].append(s)
    return [(x, y) for g in groups.values() for i, x in enumerate(g) for y in g[i + 1:]]
LEGACY_RULE_BASED = ['spp_dev', 'next_line', 'ip_stride', 'va_ampm_lite']  # summaries written before the set was recorded in them


def load_clusters(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pct(x):
    return f'{100 * x:.1f}%'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis-dir', default='results/pass2_events/analysis')
    a = ap.parse_args()

    stems = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(a.analysis_dir, '*', 'summary.json')))
    S = {s: json.load(open(os.path.join(a.analysis_dir, s, 'summary.json'))) for s in stems}
    C = {s: {k: load_clusters(os.path.join(a.analysis_dir, s, f'clusters_{k}.csv')) for k in ('py_only', 'rb_only')} for s in stems}
    sets = {tuple(S[s].get('rule_based', LEGACY_RULE_BASED)) for s in stems}
    assert len(sets) == 1, f'traces analysed with different rule-based sets: {sets}'
    RULE_BASED = list(sets.pop())
    out = []
    w = out.append

    w('# Pass 2 report: where Pythia and the rule-based prefetchers differ, access by access\n')
    regions = sorted({(S[s].get('region', {}).get('warmup_instructions', 100_000_000), S[s].get('region', {}).get('measured_instructions', 200_000_000))
                      for s in stems})
    w('Region: ' + '; '.join(f'{wu // 10**6}M warmup + {me // 10**6}M measured' for wu, me in regions) +
      ' (Ligra traces are 250M instructions long, so they use 100M + 150M as in the Pythia paper). '
      f'Rule-based = {", ".join(RULE_BASED)}. Definitions (fixed before the full runs) are in each trace\'s summary.json.\n')

    w('## 1. Verification: logging did not change the simulation\n')
    w('| trace | ' + ' | '.join(['no'] + ['pythia'] + RULE_BASED) + ' |')
    w('|---' * (len(RULE_BASED) + 3) + '|')
    for s in stems:
        cells = []
        for p in ['no', 'pythia'] + RULE_BASED:
            v = S[s]['verification_vs_pass1'][p]
            ident = v['roi_stats_identical']
            if ident is None:
                same_ipc = v['ipc_pass1'] is not None and v['ipc_pass1'] == v['ipc_pass2']
                cells.append(f"IPC {v['ipc_pass2']} ({'= pass1' if same_ipc else 'pass1 n/a' if v['ipc_pass1'] is None else 'pass1 ' + str(v['ipc_pass1'])})")
            else:
                cells.append('identical' if ident else f"DIFFERENT ({v['ipc_pass1']} vs {v['ipc_pass2']})")
        w(f'| {s} | ' + ' | '.join(cells) + ' |')
    w('')

    w('## 2. The misses (no-prefetch L2 demand misses, matched in all runs)\n')
    w('| trace | misses | matched | Pythia-only | rule-based-only | both | none | Pythia-only cycles saved | rule-only cycles saved by best rule-based |')
    w('|---' * 9 + '|')
    for s in stems:
        p, c = S[s]['population'], S[s]['categories']
        n = p['misses_matched_in_all_runs']
        w(f"| {s} | {p['nopref_misses']:,} | {n:,} ({pct(n / max(1, p['nopref_misses']))}) | {c['py_only']['count']:,} ({pct(c['py_only']['share_of_population'])}) | "
          f"{c['rb_only']['count']:,} ({pct(c['rb_only']['share_of_population'])}) | {c['both']['count']:,} | {c['none']['count']:,} ({pct(c['none']['share_of_population'])}) | "
          f"{c['py_only']['cycles_saved_by_pythia']:,} | {c['rb_only']['cycles_saved_by_best_rule_based']:,} |")
    w('')
    w('Per prefetcher (on the matched misses): caught / of which late / pollution misses (no-prefetch hit became a miss)\n')
    w('| trace | ' + ' | '.join(['pythia'] + RULE_BASED) + ' |')
    w('|---' * (len(RULE_BASED) + 2) + '|')
    for s in stems:
        pp = S[s]['per_prefetcher']
        w(f'| {s} | ' + ' | '.join(f"{pp[p]['caught']:,} / {pp[p]['caught_late']:,} / {pp[p]['pollution_misses']:,}" for p in ['pythia'] + RULE_BASED) + ' |')
    w('')

    if all('lead_median' in S[s]['per_prefetcher']['pythia'] for s in stems):
        w('Timeliness (on the matched misses each prefetcher caught): median instructions from the trigger of the covering prefetch to the access '
          '/ share of catches that were late (merged into the prefetch while in flight)\n')
        w('| trace | ' + ' | '.join(['pythia'] + RULE_BASED) + ' |')
        w('|---' * (len(RULE_BASED) + 2) + '|')
        for s in stems:
            pp = S[s]['per_prefetcher']
            lead = lambda p: f"{pp[p]['lead_median']:,.0f}" if pp[p]['lead_median'] is not None else '-'
            w(f'| {s} | ' + ' | '.join(f"{lead(p)} / {pct(pp[p]['caught_late'] / max(1, pp[p]['caught']))}" for p in ['pythia'] + RULE_BASED) + ' |')
        w('')
        w('Pythia vs each rule-based prefetcher X: misses only Pythia caught / only X caught; then, on the misses both caught, '
          'whose covering prefetch was triggered earlier (X earlier / Pythia earlier)\n')
        w('| trace | ' + ' | '.join(RULE_BASED) + ' |')
        w('|---' * (len(RULE_BASED) + 1) + '|')
        tot = {p: defaultdict(int) for p in RULE_BASED}
        for s in stems:
            cells = []
            for p in RULE_BASED:
                x = S[s]['pairwise_pythia_vs'][p]
                for k, v in x.items():
                    if isinstance(v, int):
                        tot[p][k] += v
                n = max(1, x['both_caught'])
                cells.append(f"{x['pythia_caught_this_missed']:,} / {x['this_caught_pythia_missed']:,}; "
                             f"{pct(x['both_this_triggered_earlier'] / n)} / {pct(x['both_pythia_triggered_earlier'] / n)}")
            w(f'| {s} | ' + ' | '.join(cells) + ' |')
        w('| **all traces** | ' + ' | '.join(
            f"{t['pythia_caught_this_missed']:,} / {t['this_caught_pythia_missed']:,}; {pct(t['both_this_triggered_earlier'] / max(1, t['both_caught']))} / "
            f"{pct(t['both_pythia_triggered_earlier'] / max(1, t['both_caught']))}" for t in (tot[p] for p in RULE_BASED)) + ' |')
        w('')

    w('## 3. Why each rule-based prefetcher missed the Pythia-only accesses\n')
    for s in stems:
        w(f'**{s}**')
        for p in RULE_BASED:
            d = S[s]['why_rule_based_missed_py_only'][p]
            tot = max(1, sum(d.values()))
            w(f'- {p}: ' + ', '.join(f'{k} {pct(v / tot)}' for k, v in d.items()))
        d = S[s]['why_pythia_missed_rb_only']
        tot = max(1, sum(d.values()))
        w('- (reverse) why Pythia missed the rule-based-only accesses: ' + ', '.join(f'{k} {pct(v / tot)}' for k, v in d.items()))
        w('')

    w('## 4. Patterns: top Pythia-only clusters per trace\n')
    w('Cluster = (access PC, trigger PC, delta). precision = how often "trigger PC touches line X -> access PC touches X+delta within the window" '
      'holds, over the no-prefetch L2 stream (what a prefetcher sees) and over the whole program (trace).\n')
    for s in stems:
        cl = S[s]['clusters']
        w(f"**{s}**: {cl['py_only_clusters']} clusters; top 5 cover {pct(cl['py_only_top_cluster_cumulative_share'].get('5', 0))}, "
          f"top 20 {pct(cl['py_only_top_cluster_cumulative_share'].get('20', 0))}; by rule label: {cl['py_only_access_count_by_rule_label']}; "
          f"strict window: {cl.get('py_only_access_count_by_rule_label_tight')}; either order: {cl.get('py_only_access_count_by_rule_label_either_order')}; "
          f"same-PC trigger share {pct(cl['py_only_same_pc_trigger_share'] or 0)}; deltas {cl['py_only_by_delta']}\n")
        w('| # | access PC | trigger PC | delta | count | share | precision L2 | precision program | label | strict precision (window) | either order | median distance | '
          'access PC reaching L2 | stride share program / L2 |')
        w('|---' * 14 + '|')
        for r in C[s]['py_only'][:8]:
            f = lambda k: float(r[k]) if r[k] not in ('', 'None') else None
            pl, pp_, sp, s2 = f('precision_l2'), f('precision_program'), f('target_stride_share_program'), f('target_stride_share_l2')
            w(f"| {r['rank']} | {r['access_pc']} | {r['trigger_pc']} | {r['delta']} | {int(r['count']):,} | {pct(float(r['share']))} | "
              f"{pct(pl) if pl is not None else '-'} | {pct(pp_) if pp_ is not None else '-'} | {r['rule_label'] or '-'} | "
              f"{pct(f('precision_l2_tight')) + ' (' + r['tight_window'] + ')' if f('precision_l2_tight') is not None else '-'} | "
              f"{pct(f('precision_l2_either_order')) if f('precision_l2_either_order') is not None else '-'} | {r['median_distance']} | "
              f"{pct(f('target_pc_l2_fraction')) if f('target_pc_l2_fraction') is not None else '-'} | "
              f"{pct(sp) if sp is not None else '-'} / {pct(s2) if s2 is not None else '-'} |")
        w('')

    w('## 5. Cross-input check (same program, two traces)\n')
    for x, y in input_pairs(stems):
        if x not in C or y not in C:
            continue
        key = lambda r: (r['access_pc'], r['trigger_pc'], r['delta'])
        cx = {key(r): float(r['share']) for r in C[x]['py_only']}
        cy = {key(r): float(r['share']) for r in C[y]['py_only']}
        common = sorted(set(cx) & set(cy), key=lambda k: -(cx[k] + cy[k]))
        w(f'**{x} vs {y}**: {len(common)} Pythia-only clusters appear in both; they cover {pct(sum(cx[k] for k in common))} and '
          f'{pct(sum(cy[k] for k in common))} of the Pythia-only accesses respectively.\n')
        for k in common[:8]:
            w(f'- access {k[0]} <- trigger {k[1]}, delta {k[2]}: {pct(cx[k])} / {pct(cy[k])}')
        w('')

    w('## 6. Where rule-based beats Pythia: top rule-based-only clusters\n')
    for s in stems:
        w(f'**{s}**')
        for r in C[s]['rb_only'][:5]:
            w(f"- {r['caught_by']}: access {r['access_pc']} <- trigger {r['trigger_pc']}, delta {r['delta']}: {int(r['count']):,} "
              f"({pct(float(r['share']))}), label {r['rule_label'] or '-'}, Pythia: {r['why_missed']}")
        w('')

    path = os.path.join(a.analysis_dir, 'REPORT.md')
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print(f'wrote {path} ({len(out)} lines, {len(stems)} traces)')


if __name__ == '__main__':
    main()
