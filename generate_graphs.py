#!/usr/bin/env python3
"""
Generate comprehensive, presentation-quality ChampSim comparison plots.

Filename convention in results/:
    <TRACE>_<CACHE>_<PREFETCHER>_<ADDRESS>_<CORES>_<WARMUP>_<SIM>.txt

Generated Plots in graphs/:
    1. Per-Trace Comparison Plots:
       - IPC comparison for each trace across all prefetchers (Physical vs Virtual)
       - Prefetcher accuracy comparison for each trace across prefetchers (Physical vs Virtual)
    2. Per-Prefetcher Comparison Plots:
       - IPC comparison across all 4 traces for each prefetcher (Physical vs Virtual)
       - Prefetcher accuracy comparison across all 4 traces for each prefetcher (Physical vs Virtual)
    3. Multi-Panel Overview / Summary Grids:
       - IPC 2x2 grid overview across all benchmark traces
       - Accuracy 2x2 grid overview across all benchmark traces
       - IPC multi-panel overview across all prefetchers
       - Accuracy multi-panel overview across all prefetchers
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure writable matplotlib config dir
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


PARAMS = ("trace", "cache", "prefetcher", "address", "cores", "warmup", "sim")
DISPLAY = {
    "trace": "Trace",
    "cache": "Cache",
    "prefetcher": "Prefetcher",
    "address": "Address",
    "cores": "Cores",
    "warmup": "Warmup",
    "sim": "Simulation",
}

PREFETCHER_ORDER = [
    "no",
    "nextline",
    "ipstride",
    "sppdev",
    "vaampmlite",
    "pythia",
    "pathfinder",
]

# Presentation-ready color palette matching graph_plots.py
PALETTE = [
    "#2F5597",  # Deep Steel Blue (Physical)
    "#70AD47",  # Vivid Olive Green (Virtual)
    "#ED7D31",  # Warm Amber / Orange
    "#5B9BD5",  # Light Blue
    "#A5A5A5",  # Medium Grey
    "#FFC000",  # Golden Yellow
    "#4472C4",  # Royal Blue
    "#548235",  # Forest Green
    "#C55A11",  # Rust Red
    "#7F7F7F",  # Charcoal Grey
]


@dataclass(frozen=True)
class Config:
    trace: str
    cache: str
    prefetcher: str
    address: str
    cores: str
    warmup: str
    sim: str


@dataclass
class Result:
    config: Config
    path: Path
    ipc: Optional[float]
    requested: Optional[int]
    issued: Optional[int]
    useful: Optional[int]
    useless: Optional[int]

    @property
    def accuracy(self) -> Optional[float]:
        if self.issued is None or self.useful is None or self.issued == 0:
            return None
        return 100.0 * self.useful / self.issued


def parse_filename(path: Path) -> Config:
    """Parse underscore-separated fields from right to left."""
    stem = path.name
    while stem.endswith(".txt"):
        stem = stem[:-4]

    parts = stem.rsplit("_", 6)
    if len(parts) != 7:
        raise ValueError(
            f"expected at least 7 underscore-separated fields, got {len(parts)}"
        )

    trace, cache, prefetcher, address, cores, warmup, sim = parts
    if not all(parts):
        raise ValueError("empty filename field")

    return Config(trace, cache, prefetcher, address, cores, warmup, sim)


def parse_result(path: Path, config: Config) -> Result:
    text = path.read_text(errors="replace")

    ipc_matches = re.findall(
        r"CPU\s+(\d+)\s+cumulative IPC:\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s+instructions:\s*(\d+)"
        r"\s+cycles:\s*(\d+)",
        text,
    )
    if not ipc_matches:
        raise ValueError("simulation IPC not found")

    cpu_ipc = {}
    for cpu, ipc_str, instr, cycles in ipc_matches:
        cpu_ipc[int(cpu)] = (float(ipc_str), int(instr), int(cycles))

    if not cpu_ipc:
        raise ValueError("simulation IPC not found")

    ipc = sum(v[0] for v in cpu_ipc.values())

    requested = issued = useful = useless = None

    if config.prefetcher.lower() != "no":
        cache_name = config.cache
        if cache_name.upper() == "NONE":
            raise ValueError(
                "prefetcher is not 'no' but cache is NONE; "
                "cannot determine prefetcher installation cache"
            )

        cache_pat = re.compile(
            rf"cpu\d+_?{re.escape(cache_name)}\s+PREFETCH\s+"
            rf"REQUESTED:\s*(\d+)\s+ISSUED:\s*(\d+)\s+"
            rf"USEFUL:\s*(\d+)\s+USELESS:\s*(\d+)",
            re.IGNORECASE,
        )
        m = cache_pat.search(text)
        if not m:
            cache_pat = re.compile(
                rf"cpu\d+->cpu\d+_{re.escape(cache_name)}\s+PREFETCH\s+"
                rf"REQUESTED:\s*(\d+)\s+ISSUED:\s*(\d+)\s+"
                rf"USEFUL:\s*(\d+)\s+USELESS:\s*(\d+)",
                re.IGNORECASE,
            )
            m = cache_pat.search(text)

        if not m:
            raise ValueError(
                f"PREFETCH REQUESTED/ISSUED/USEFUL/USELESS line "
                f"not found for cache {cache_name}"
            )

        requested, issued, useful, useless = map(int, m.groups())

    return Result(
        config=config,
        path=path,
        ipc=ipc,
        requested=requested,
        issued=issued,
        useful=useful,
        useless=useless,
    )


def discover_results(results_dir: Path, args: Optional[argparse.Namespace] = None) -> Tuple[List[Result], List[str]]:
    results = []
    errors = []

    for path in sorted(results_dir.iterdir()):
        if not path.is_file() or not path.name.endswith(".txt"):
            continue

        try:
            config = parse_filename(path)
        except ValueError:
            continue

        dummy = Result(config, path, None, None, None, None, None)
        if args is not None and not fixed_match(dummy, args):
            continue

        try:
            results.append(parse_result(path, config))
        except ValueError as e:
            errors.append(f"{path.name}: {e}")

    return results, errors


def fixed_match(result: Result, args: argparse.Namespace) -> bool:
    for p in PARAMS:
        requested = getattr(args, p)
        if requested == "none":
            continue

        actual = getattr(result.config, p)

        if (
            p == "cache"
            and args.prefetcher == "none"
            and result.config.prefetcher.lower() == "no"
            and result.config.cache.upper() == "NONE"
        ):
            continue

        if actual.lower() != requested.lower():
            return False

    return True


def variable_params(args: argparse.Namespace) -> List[str]:
    return [p for p in PARAMS if getattr(args, p) == "none"]


def validate_semantics(args: argparse.Namespace) -> List[str]:
    errors = []
    vars_ = variable_params(args)

    if len(vars_) > 3:
        errors.append(
            "At most 3 variable parameters are allowed. "
            f"Found {len(vars_)}: {', '.join(DISPLAY[p] for p in vars_)}"
        )

    if args.prefetcher.lower() == "no" and args.cache.lower() != "none":
        errors.append(
            "CACHE is not meaningful when prefetcher=no. "
            "Use cache=NONE or cache=none (if varying)."
        )

    if args.prefetcher.lower() != "no" and args.cache != "none" and args.cache.upper() == "NONE":
        errors.append(
            "CACHE=NONE is only valid for the no-prefetcher configuration."
        )

    return errors


def sort_key(param: str, value: str):
    if param == "prefetcher":
        v_low = value.lower()
        if v_low in PREFETCHER_ORDER:
            return (0, PREFETCHER_ORDER.index(v_low))
        return (1, v_low)
    if param == "address":
        if value.lower() == "physical":
            return (0, 0)
        if value.lower() == "virtual":
            return (0, 1)
        return (1, value.lower())
    m = re.fullmatch(r"(\d+)(.*)", value)
    if m:
        return (0, int(m.group(1)), m.group(2).lower())
    return (1, value.lower())


def ordered_values(results: Sequence[Result], param: str) -> List[str]:
    vals = {getattr(r.config, param) for r in results}
    return sorted(vals, key=lambda v: sort_key(param, v))


def pretty_value(param: str, value: str) -> str:
    if param == "cores":
        return value.replace("core", " core")
    if param == "warmup":
        return value.replace("warm", " warm")
    if param == "sim":
        return value.replace("sim", " sim")
    if param == "prefetcher" and value.lower() == "no":
        return "No prefetcher"
    if param == "cache" and value.upper() == "NONE":
        return "N/A"
    return value


def safe_token(value: str) -> str:
    value = value.replace("/", "-").replace(" ", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def experiment_tag(args: argparse.Namespace) -> str:
    parts = []
    for p in PARAMS:
        value = getattr(args, p)
        label = "VAR" if value == "none" else value
        parts.append(f"{p}-{safe_token(label)}")
    return "_".join(parts)


def metric_value(r: Result, metric: str) -> Optional[float]:
    return r.ipc if metric == "IPC" else r.accuracy


def metric_label(metric: str) -> str:
    return "IPC" if metric == "IPC" else "Prefetcher accuracy (%)"


def metric_filename(metric: str) -> str:
    return "ipc" if metric == "IPC" else "accuracy"


def choose_colors(labels: Sequence[str]) -> Dict[str, str]:
    # Fixed mappings for physical/virtual address modes for visual consistency
    color_map = {}
    for i, label in enumerate(labels):
        l_low = label.lower()
        if l_low == "physical":
            color_map[label] = PALETTE[0]  # Deep Blue
        elif l_low == "virtual":
            color_map[label] = PALETTE[1]  # Green
        else:
            color_map[label] = PALETTE[i % len(PALETTE)]
    return color_map


def add_value_labels(ax, bars, fmt, rotation: int = 0, fontsize: float = 8.0):
    for bar in bars:
        height = bar.get_height()
        if not math.isfinite(height) or height < 0:
            continue
        ax.annotate(
            fmt(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="semibold",
            rotation=rotation,
        )


def style_axis(ax, title: str, subtitle: str, ylabel: str):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16, loc="left")
    ax.text(
        0, 1.015, subtitle,
        transform=ax.transAxes,
        fontsize=9.0,
        color="#555555",
        va="bottom",
    )
    ax.set_ylabel(ylabel, fontsize=10.5, fontweight="500")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis="both", labelsize=9)


def save_fig(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fixed_description(args: argparse.Namespace) -> str:
    parts = []
    for p in PARAMS:
        value = getattr(args, p)
        if value.lower() != "none":
            parts.append(f"{DISPLAY[p]}={pretty_value(p, value)}")
    return ", ".join(parts) if parts else "all parameters vary"


# ============================================================
# CORE PLOTTING FUNCTIONS
# ============================================================

def plot_two_variables(results, vars_, metric, args, out_dir: Path, custom_filename: Optional[str] = None):
    a, b = vars_

    if b == "address":
        group_param, bar_param = a, b
    elif a == "address":
        group_param, bar_param = b, a
    else:
        group_param, bar_param = a, b

    groups = ordered_values(results, group_param)
    bars = ordered_values(results, bar_param)

    lookup = {
        (getattr(r.config, group_param), getattr(r.config, bar_param)): r
        for r in results
    }

    width = 0.72 / max(1, len(bars))
    x = list(range(len(groups)))

    fig, ax = plt.subplots(figsize=(max(8.5, 1.35 * len(groups)), 6.0))
    colors = choose_colors([pretty_value(bar_param, v) for v in bars])
    all_bars = []

    for j, bv in enumerate(bars):
        for i, gv in enumerate(groups):
            r = lookup.get((gv, bv))
            y = metric_value(r, metric) if r else None
            if y is None:
                continue

            pos = i + (j - (len(bars) - 1) / 2) * width
            bar = ax.bar(
                pos, y, width=width * 0.88,
                color=colors[pretty_value(bar_param, bv)],
                label=pretty_value(bar_param, bv) if i == 0 else None,
                edgecolor="white",
                linewidth=0.8,
            )[0]
            all_bars.append(bar)

    ax.set_xticks(x)
    ax.set_xticklabels([pretty_value(group_param, v) for v in groups], fontweight="500")
    ax.set_xlabel(DISPLAY[group_param], fontsize=11, fontweight="bold", labelpad=8)

    title = f"{metric_label(metric)}: {DISPLAY[group_param]} Comparison by {DISPLAY[bar_param]}"
    subtitle = fixed_description(args)
    style_axis(ax, title, subtitle, metric_label(metric))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            frameon=False,
            ncols=min(3, len(labels)),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            fontsize=9.5,
        )

    if metric == "accuracy":
        ax.set_ylim(0, 100)
    elif metric == "IPC" and all_bars:
        max_val = max(b.get_height() for b in all_bars)
        ax.set_ylim(0, max_val * 1.15)

    if all_bars:
        fmt = (lambda val: f"{val:.3f}" if val < 1.0 else f"{val:.2f}") if metric == "IPC" else (lambda val: f"{val:.1f}%")
        add_value_labels(ax, all_bars, fmt)

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    
    if custom_filename:
        filename = out_dir / custom_filename
    else:
        filename = out_dir / f"{metric_filename(metric)}_{experiment_tag(args)}_vs_{group_param}_by_{bar_param}.png"
    
    save_fig(fig, filename)
    return filename


def plot_summary_grid(
    results: Sequence[Result],
    facet_param: str,
    group_param: str,
    bar_param: str,
    metric: str,
    args: argparse.Namespace,
    out_dir: Path,
    custom_filename: str,
    title_override: Optional[str] = None,
):
    facet_vals = ordered_values(results, facet_param)
    if facet_param == "cache":
        facet_vals = [v for v in facet_vals if v.upper() != "NONE"]

    group_vals = ordered_values(results, group_param)
    bar_vals = ordered_values(results, bar_param)

    n = len(facet_vals)
    ncols = 2 if n >= 4 else min(3, max(1, n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5 * ncols, 4.8 * nrows),
        squeeze=False,
    )
    axes_flat = [ax for row in axes for ax in row]

    colors = choose_colors([pretty_value(bar_param, v) for v in bar_vals])

    lookup = {
        (
            getattr(r.config, facet_param),
            getattr(r.config, group_param),
            getattr(r.config, bar_param),
        ): r
        for r in results
    }

    # Baseline lookup for prefetcher=no
    baseline_lookup = {
        (getattr(r.config, facet_param), getattr(r.config, bar_param)): r
        for r in results
        if r.config.prefetcher.lower() == "no"
    }

    width = 0.76 / max(1, len(bar_vals))

    for idx, fv in enumerate(facet_vals):
        ax = axes_flat[idx]
        subplot_bars = []

        for j, bv in enumerate(bar_vals):
            for i, gv in enumerate(group_vals):
                r = lookup.get((fv, gv, bv))
                if r is None and gv.lower() == "no" and group_param == "prefetcher":
                    r = baseline_lookup.get((fv, bv))

                y = metric_value(r, metric) if r else None
                if y is None:
                    continue

                pos = i + (j - (len(bar_vals) - 1) / 2) * width
                bar = ax.bar(
                    pos, y, width=width * 0.88,
                    color=colors[pretty_value(bar_param, bv)],
                    label=pretty_value(bar_param, bv) if (idx == 0 and i == 0) else None,
                    edgecolor="white",
                    linewidth=0.6,
                )[0]
                subplot_bars.append(bar)

        if subplot_bars:
            fmt = (lambda val: f"{val:.3f}" if val < 1.0 else f"{val:.2f}") if metric == "IPC" else (lambda val: f"{val:.1f}%")
            add_value_labels(ax, subplot_bars, fmt, rotation=0, fontsize=7.5)

        ax.set_title(
            f"{DISPLAY[facet_param]}: {pretty_value(facet_param, fv)}",
            fontsize=12, fontweight="bold", loc="left", pad=10,
        )
        ax.set_xticks(range(len(group_vals)))
        ax.set_xticklabels([pretty_value(group_param, v) for v in group_vals], fontsize=8.5, rotation=15 if len(group_vals) > 5 else 0)
        ax.set_xlabel(DISPLAY[group_param], fontsize=9.5, fontweight="bold")
        ax.set_ylabel(metric_label(metric), fontsize=9.5)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.35)
        ax.spines["bottom"].set_alpha(0.35)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(labelsize=8)

        if metric == "accuracy":
            ax.set_ylim(0, 100)
        elif metric == "IPC" and subplot_bars:
            max_val = max(b.get_height() for b in subplot_bars)
            ax.set_ylim(0, max_val * 1.18)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=min(4, len(labels)),
            fontsize=10,
        )

    main_title = title_override or f"{metric_label(metric)} Comparison Grid"
    sub_title = (
        f"Multi-panel view by {DISPLAY[facet_param]}; Grouped by {DISPLAY[group_param]}; "
        f"Compared across {DISPLAY[bar_param]} ({', '.join(pretty_value(bar_param, b) for b in bar_vals)}). "
        f"{fixed_description(args)}"
    )

    fig.text(0.05, 0.98, main_title, fontsize=16, fontweight="bold", ha="left", va="top")
    fig.text(0.05, 0.955, sub_title, fontsize=9.0, color="#555555", ha="left", va="top")

    fig.subplots_adjust(
        left=0.06, right=0.98, bottom=0.08,
        top=0.88, hspace=0.38, wspace=0.22,
    )

    filename = out_dir / custom_filename
    save_fig(fig, filename)
    return filename


# ============================================================
# BATCH GENERATION WORKFLOW
# ============================================================

def generate_all_graphs(results_dir: Path = Path("results"), out_dir: Path = Path("graphs")):
    """Generate the complete set of standard and overview graphs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results, errors = discover_results(results_dir)

    if errors:
        print("Note: Ignored non-standard files in results/:")
        for e in errors:
            print(f"  - {e}")

    print(f"\nDiscovered {len(all_results)} valid result entries from {results_dir.resolve()}.")

    traces = ["600.perlbench_s-210B", "602.gcc_s-1850B", "605.mcf_s-1152B", "619.lbm_s-2676B"]
    prefetchers = ["nextline", "ipstride", "sppdev", "vaampmlite", "pythia", "pathfinder"]

    generated_files = []

    print("\n--- [1/3] Generating Per-Trace Comparison Graphs (Physical vs Virtual) ---")
    for tr in traces:
        args = argparse.Namespace(
            trace=tr,
            cache="L2C",
            prefetcher="none",
            address="none",
            cores="1core",
            warmup="100Kwarm",
            sim="1Msim",
            metric="both",
            results_dir=str(results_dir),
            plots_dir=str(out_dir),
        )
        selected = [r for r in all_results if fixed_match(r, args)]
        
        # 1. IPC
        valid_ipc = [r for r in selected if r.ipc is not None]
        if valid_ipc:
            fn = plot_two_variables(
                valid_ipc, ["prefetcher", "address"], "IPC", args, out_dir,
                custom_filename=f"ipc_trace_{tr}_prefetchers_by_address.png",
            )
            # Also write standard tag name
            plot_two_variables(valid_ipc, ["prefetcher", "address"], "IPC", args, out_dir)
            generated_files.append(fn)
            print(f"  [+] Generated: {fn.name}")

        # 2. Accuracy
        valid_acc = [r for r in selected if r.accuracy is not None]
        if valid_acc:
            fn = plot_two_variables(
                valid_acc, ["prefetcher", "address"], "accuracy", args, out_dir,
                custom_filename=f"accuracy_trace_{tr}_prefetchers_by_address.png",
            )
            # Also write standard tag name
            plot_two_variables(valid_acc, ["prefetcher", "address"], "accuracy", args, out_dir)
            generated_files.append(fn)
            print(f"  [+] Generated: {fn.name}")

    print("\n--- [2/3] Generating Per-Prefetcher Comparison Graphs (Across Traces) ---")
    all_prefs_to_plot = ["no"] + prefetchers
    for pf in all_prefs_to_plot:
        cache_val = "NONE" if pf == "no" else "L2C"
        args = argparse.Namespace(
            trace="none",
            cache=cache_val,
            prefetcher=pf,
            address="none",
            cores="1core",
            warmup="100Kwarm",
            sim="1Msim",
            metric="both",
            results_dir=str(results_dir),
            plots_dir=str(out_dir),
        )
        selected = [r for r in all_results if fixed_match(r, args)]

        # 1. IPC
        valid_ipc = [r for r in selected if r.ipc is not None]
        if valid_ipc:
            fn = plot_two_variables(
                valid_ipc, ["trace", "address"], "IPC", args, out_dir,
                custom_filename=f"ipc_prefetcher_{pf}_traces_by_address.png",
            )
            plot_two_variables(valid_ipc, ["trace", "address"], "IPC", args, out_dir)
            generated_files.append(fn)
            print(f"  [+] Generated: {fn.name}")

        # 2. Accuracy (skip 'no' baseline)
        if pf != "no":
            valid_acc = [r for r in selected if r.accuracy is not None]
            if valid_acc:
                fn = plot_two_variables(
                    valid_acc, ["trace", "address"], "accuracy", args, out_dir,
                    custom_filename=f"accuracy_prefetcher_{pf}_traces_by_address.png",
                )
                plot_two_variables(valid_acc, ["trace", "address"], "accuracy", args, out_dir)
                generated_files.append(fn)
                print(f"  [+] Generated: {fn.name}")

    print("\n--- [3/3] Generating Overview / Summary Multi-Panel Grids ---")
    summary_args = argparse.Namespace(
        trace="none",
        cache="L2C",
        prefetcher="none",
        address="none",
        cores="1core",
        warmup="100Kwarm",
        sim="1Msim",
        metric="both",
        results_dir=str(results_dir),
        plots_dir=str(out_dir),
    )
    selected_l2c = [
        r for r in all_results
        if r.config.cores == "1core"
        and r.config.warmup == "100Kwarm"
        and r.config.sim == "1Msim"
        and (r.config.cache == "L2C" or (r.config.prefetcher.lower() == "no" and r.config.cache.upper() == "NONE"))
    ]

    # Grid 1: IPC by Trace (Facet=Trace, Group=Prefetcher, Bar=Address)
    valid_ipc = [r for r in selected_l2c if r.ipc is not None]
    if valid_ipc:
        fn = plot_summary_grid(
            valid_ipc,
            facet_param="trace",
            group_param="prefetcher",
            bar_param="address",
            metric="IPC",
            args=summary_args,
            out_dir=out_dir,
            custom_filename="ipc_all_traces_summary_grid.png",
            title_override="IPC Comparison Across All Benchmarks (L2C Cache)",
        )
        generated_files.append(fn)
        print(f"  [+] Generated: {fn.name}")

    # Grid 2: Accuracy by Trace (Facet=Trace, Group=Prefetcher, Bar=Address)
    valid_acc = [r for r in selected_l2c if r.accuracy is not None and r.config.prefetcher.lower() != "no"]
    if valid_acc:
        fn = plot_summary_grid(
            valid_acc,
            facet_param="trace",
            group_param="prefetcher",
            bar_param="address",
            metric="accuracy",
            args=summary_args,
            out_dir=out_dir,
            custom_filename="accuracy_all_traces_summary_grid.png",
            title_override="Prefetcher Accuracy (%) Across All Benchmarks (L2C Cache)",
        )
        generated_files.append(fn)
        print(f"  [+] Generated: {fn.name}")

    # Grid 3: IPC by Prefetcher (Facet=Prefetcher, Group=Trace, Bar=Address)
    if valid_ipc:
        fn = plot_summary_grid(
            valid_ipc,
            facet_param="prefetcher",
            group_param="trace",
            bar_param="address",
            metric="IPC",
            args=summary_args,
            out_dir=out_dir,
            custom_filename="ipc_all_prefetchers_summary_grid.png",
            title_override="IPC Comparison Across All Prefetchers (L2C Cache)",
        )
        generated_files.append(fn)
        print(f"  [+] Generated: {fn.name}")

    # Grid 4: Accuracy by Prefetcher (Facet=Prefetcher, Group=Trace, Bar=Address)
    if valid_acc:
        fn = plot_summary_grid(
            valid_acc,
            facet_param="prefetcher",
            group_param="trace",
            bar_param="address",
            metric="accuracy",
            args=summary_args,
            out_dir=out_dir,
            custom_filename="accuracy_all_prefetchers_summary_grid.png",
            title_override="Prefetcher Accuracy (%) Across All Prefetchers (L2C Cache)",
        )
        generated_files.append(fn)
        print(f"  [+] Generated: {fn.name}")

    print(f"\nSuccessfully generated {len(generated_files)} high-resolution plots in '{out_dir.resolve()}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ChampSim comparison bar plots from result files."
    )
    for name in PARAMS:
        parser.add_argument(
            f"--{name}",
            default=None,
            help=f"{DISPLAY[name]} value, or 'none' to make it a variable.",
        )
    parser.add_argument(
        "--metric",
        choices=("IPC", "accuracy", "both"),
        default="both",
        help="Metric to plot. 'both' creates separate IPC and accuracy plots.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing ChampSim output files (default: results).",
    )
    parser.add_argument(
        "--plots-dir",
        default="graphs",
        help="Directory where generated plots are written (default: graphs).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all standard comparison plots and overview grids.",
    )

    args = parser.parse_args()

    # If any specific param is provided, run targeted query; otherwise run full batch
    any_param_set = any(getattr(args, p) is not None for p in PARAMS)
    if not any_param_set or args.all:
        generate_all_graphs(Path(args.results_dir), Path(args.plots_dir))
    else:
        # Fill defaults for unset parameters
        for p in PARAMS:
            if getattr(args, p) is None:
                setattr(args, p, "none")
        
        errors = validate_semantics(args)
        if errors:
            for e in errors:
                print(f"ERROR: {e}")
            sys.exit(1)

        results_dir = Path(args.results_dir)
        out_dir = Path(args.plots_dir)
        all_results, _ = discover_results(results_dir, args)
        vars_ = variable_params(args)

        metrics = ("IPC", "accuracy") if args.metric == "both" else (args.metric,)
        for m in metrics:
            valid = [r for r in all_results if metric_value(r, m) is not None]
            if len(vars_) == 2:
                plot_two_variables(valid, vars_, m, args, out_dir)
            else:
                print(f"Note: Standard 2-variable plotting recommended for {vars_}.")


if __name__ == "__main__":
    main()

