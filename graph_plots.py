"""
Parse ChampSim result files and generate professional comparison plots.

Filename convention:
    <TRACE>_<CACHE>_<PREFETCHER>_<ADDRESS>_<CORES>_<WARMUP>_<SIM>.txt

The filename is parsed from right to left so TRACE may contain underscores.

Examples:
    python3 plot_champsim.py --trace perlbench_210B --cache none \
        --prefetcher nextline --address none --cores 1core \
        --warmup 100Kwarm --sim 1Msim --metric IPC

    python3 plot_champsim.py --trace perlbench_210B --cache none \
        --prefetcher nextline --address none --cores 1core \
        --warmup 100Kwarm --sim 1Msim --metric both
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

# Restrained, presentation-friendly palette. The exact colors are assigned
# consistently to categories within a single figure.
PALETTE = [
    "#2F5597", "#70AD47", "#ED7D31", "#5B9BD5", "#A5A5A5",
    "#FFC000", "#4472C4", "#548235", "#C55A11", "#7F7F7F",
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
        # Current plotting definition requested by the user:
        # useful issued prefetches / issued prefetches.
        if self.issued is None or self.useful is None or self.issued == 0:
            return None
        return 100.0 * self.useful / self.issued


def parse_filename(path: Path) -> Config:
    """Parse exactly six right-most underscore-separated fields."""
    stem = path.name
    # Be tolerant of accidental repeated .txt suffixes in result filenames
    # (e.g. the supplied *_1Msim.txt.txt example).
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

    # Use the Region-of-Interest/simulation IPC, not warmup IPC. ChampSim
    # prints one cumulative IPC line per CPU. For multicore runs, report the
    # aggregate IPC (sum of the per-CPU IPCs) rather than accidentally taking
    # only the last CPU printed.
    ipc_matches = re.findall(
        r"CPU\s+(\d+)\s+cumulative IPC:\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s+instructions:\s*(\d+)"
        r"\s+cycles:\s*(\d+)",
        text,
    )
    if not ipc_matches:
        raise ValueError("simulation IPC not found")

    cpu_ipc={}
    for cpu,ipc_str,instr,cycles in ipc_matches:
        cpu_ipc[int(cpu)]=(float(ipc_str),int(instr),int(cycles))

    if not cpu_ipc:
        raise ValueError("simulation IPC not found")

    # The last occurrence for each CPU is the final simulation/ROI value.
    ipc=sum(v[0] for v in cpu_ipc.values())

    # Find the cache named by the filename. For prefetcher != no, the
    # prefetch statistics must come from the cache where it is installed.
    requested = issued = useful = useless = None

    if config.prefetcher.lower() != "no":
        cache_name = config.cache
        if cache_name.upper() == "NONE":
            raise ValueError(
                "prefetcher is not 'no' but cache is NONE; "
                "cannot determine prefetcher installation cache"
            )

        # Cache-stat block, then its PREFETCH REQUESTED line.
        cache_pat = re.compile(
            rf"cpu\d+_?{re.escape(cache_name)}\s+PREFETCH\s+"
            rf"REQUESTED:\s*(\d+)\s+ISSUED:\s*(\d+)\s+"
            rf"USEFUL:\s*(\d+)\s+USELESS:\s*(\d+)",
            re.IGNORECASE,
        )
        m = cache_pat.search(text)
        if not m:
            # ChampSim output in the supplied files uses cpu0_L1D/L2C.
            # Keep a fallback accepting a ->cache style prefix.
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


def discover_results(results_dir: Path, args: argparse.Namespace) -> Tuple[List[Result], List[str]]:
    """Read only result files that can match the current run_plots() call.

    Filename parsing is cheap and is done for every .txt file. The expensive
    content parsing and its warnings happen only for files relevant to this
    requested experiment.
    """
    results=[]
    errors=[]

    for path in sorted(results_dir.iterdir()):
        if not path.is_file() or not path.name.endswith(".txt"):
            continue

        try:
            config=parse_filename(path)
        except ValueError:
            # A malformed/unrelated filename cannot be matched reliably.
            # Ignore it here rather than emitting noise for every run.
            continue

        # Construct a lightweight placeholder only for matching the filename.
        # No file contents are read until after this test succeeds.
        dummy=Result(config,path,None,None,None,None,None)
        if not fixed_match(dummy,args):
            continue

        try:
            results.append(parse_result(path,config))
        except ValueError as e:
            errors.append(f"{path.name}: {e}")

    return results,errors


def normalize_arg(value: str) -> str:
    return value.strip()


def fixed_match(result: Result, args: argparse.Namespace) -> bool:
    for p in PARAMS:
        requested=getattr(args,p)
        if requested=="none":
            continue

        actual=getattr(result.config,p)

        # A no-prefetcher result has CACHE=NONE because there is no
        # prefetcher placement. When PREFETCHER is being varied and CACHE is
        # fixed to a real cache, that no-prefetcher result is still a valid
        # baseline for the comparison.
        if (
            p=="cache"
            and args.prefetcher=="none"
            and result.config.prefetcher.lower()=="no"
            and result.config.cache.upper()=="NONE"
        ):
            continue

        if actual.lower()!=requested.lower():
            return False

    return True


def variable_params(args: argparse.Namespace) -> List[str]:
    # Exact lowercase "none" is the special command-line marker meaning
    # "vary this parameter". This leaves uppercase "NONE" available as the
    # literal cache value used by no-prefetcher result files.
    return [p for p in PARAMS if getattr(args, p) == "none"]


def validate_semantics(args: argparse.Namespace) -> List[str]:
    errors = []
    vars_ = variable_params(args)

    if len(vars_) > 3:
        errors.append(
            "At most 3 variable parameters are allowed. "
            f"Found {len(vars_)}: {', '.join(DISPLAY[p] for p in vars_)}"
        )

    # Cache=NONE is meaningful only for the no-prefetcher configuration.
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


def sort_key(value: str):
    m = re.fullmatch(r"(\d+)(.*)", value)
    if m:
        return (0, int(m.group(1)), m.group(2).lower())
    return (1, value.lower())


def ordered_values(results: Sequence[Result], param: str) -> List[str]:
    vals = {getattr(r.config, param) for r in results}
    return sorted(vals, key=sort_key)


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


def is_cache_na(result: Result) -> bool:
    return result.config.prefetcher.lower() == "no" and result.config.cache.upper() == "NONE"


def validate_dataset(results: Sequence[Result], args: argparse.Namespace) -> Tuple[List[Result], List[str]]:
    selected = [r for r in results if fixed_match(r, args)]
    warnings: List[str] = []

    if not selected:
        return selected, ["No result files match the fixed parameters."]

    vars_ = variable_params(args)

    # A cache value is not a real comparison dimension for a no-prefetcher row.
    # If cache is variable and prefetcher is fixed to no, reject it earlier.
    if "cache" in vars_ and args.prefetcher.lower() == "no":
        return [], [
            "CACHE cannot be varied when prefetcher=no because there is no "
            "prefetcher placement."
        ]

    # Warn about malformed/incomplete combinations rather than silently
    # converting missing measurements into zero.
    if vars_:
        expected=set()
        for combo in _cartesian_values(selected,vars_):
            combo_dict=dict(zip(vars_,combo))

            # CACHE=NONE is the special no-prefetcher configuration. It is
            # not a normal cache-placement choice, so do not report invalid
            # cache/prefetcher pairs as missing data.
            if "cache" in combo_dict and "prefetcher" in combo_dict:
                c=combo_dict["cache"]
                pf=combo_dict["prefetcher"]
                if pf.lower()=="no" and c.upper()!="NONE":
                    continue
                if pf.lower()!="no" and c.upper()=="NONE":
                    continue

            # A no-prefetcher baseline is intentionally not required for every
            # address value. Its cache placement is N/A, and the common
            # baseline file may exist only for the physical mapping. Do not
            # call that absence a missing prefetcher experiment.
            if "prefetcher" in combo_dict and combo_dict["prefetcher"].lower()=="no":
                continue

            expected.add(combo)

        actual={
            tuple(getattr(r.config,p) for p in vars_)
            for r in selected
        }

        missing=sorted(
            expected-actual,
            key=lambda x:tuple(sort_key(v) for v in x),
        )
        if missing:
            warnings.append("Missing result combinations:")
            for combo in missing:
                warnings.append(
                    "  " + ", ".join(
                        f"{DISPLAY[p]}={pretty_value(p, v)}"
                        for p, v in zip(vars_, combo)
                    )
                )

    return selected, warnings


def _cartesian_values(results: Sequence[Result], vars_: Sequence[str]):
    vals = [ordered_values(results, p) for p in vars_]
    if not vals:
        return []
    out = [()]
    for choices in vals:
        out = [x + (v,) for x in out for v in choices]
    return out


def combinations_for(results: Sequence[Result], vars_: Sequence[str]):
    if not vars_:
        return [()]
    return _cartesian_values(results, vars_)


def result_map(results: Sequence[Result]) -> Dict[Tuple[str, ...], Result]:
    return {
        tuple(getattr(r.config, p) for p in PARAMS): r
        for r in results
    }


def fixed_description(args: argparse.Namespace) -> str:
    parts = []
    for p in PARAMS:
        value = getattr(args, p)
        if value.lower() != "none":
            parts.append(f"{DISPLAY[p]}={pretty_value(p, value)}")
    return ", ".join(parts) if parts else "all filename parameters vary"


def safe_token(value: str) -> str:
    """Make an argument value safe for use in a filename."""
    value=value.replace("/","-").replace(" ","-")
    return re.sub(r"[^A-Za-z0-9_.-]+","-",value)


def experiment_tag(args: argparse.Namespace) -> str:
    """Create a unique, readable tag from the run_plots() arguments."""
    parts=[]
    for p in PARAMS:
        value=getattr(args,p)
        label="VAR" if value=="none" else value
        parts.append(f"{p}-{safe_token(label)}")
    return "_".join(parts)

def metric_value(r: Result, metric: str) -> Optional[float]:
    return r.ipc if metric == "IPC" else r.accuracy


def metric_label(metric: str) -> str:
    return "IPC" if metric == "IPC" else "Prefetcher accuracy (%)"


def metric_filename(metric: str) -> str:
    return "ipc" if metric == "IPC" else "accuracy"


def choose_colors(labels: Sequence[str]) -> Dict[str, str]:
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}


def add_value_labels(ax, bars, fmt):
    for bar in bars:
        height = bar.get_height()
        if not math.isfinite(height):
            continue
        ax.annotate(
            fmt(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="normal",
        )


def style_axis(ax, title: str, subtitle: str, ylabel: str):
    ax.set_title(title, fontsize=15, fontweight="bold", pad=16, loc="left")
    ax.text(
        0, 1.01, subtitle,
        transform=ax.transAxes,
        fontsize=9.5,
        color="#666666",
        va="bottom",
    )
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.25)
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


def plot_one_variable(results, var, metric, args, out_dir):
    values = ordered_values(results, var)
    lookup = {getattr(r.config, var): r for r in results}

    xs = list(range(len(values)))
    ys = [metric_value(lookup[v], metric) for v in values]

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(values)), 5.8))
    bars = []
    for i, y in enumerate(ys):
        if y is not None:
            bars.append(ax.bar(i, y, width=0.62, color=PALETTE[0], label=metric)[0])

    ax.set_xticks(xs)
    ax.set_xticklabels([pretty_value(var, v) for v in values])
    ax.set_xlabel(DISPLAY[var])
    title = f"{metric_label(metric)} by {DISPLAY[var]}"
    subtitle = fixed_description(args)
    style_axis(ax, title, subtitle, metric_label(metric))

    if metric == "accuracy":
        ax.set_ylim(0, min(100, max(100, max((y or 0) for y in ys) * 1.18)))

    if bars:
        fmt = (lambda x: f"{x:.2f}") if metric == "IPC" else (lambda x: f"{x:.1f}%")
        add_value_labels(ax, bars, fmt)

    fig.tight_layout()
    filename = out_dir / f"{metric_filename(metric)}_{experiment_tag(args)}_vs_{var}.png"
    save_fig(fig, filename)


def plot_two_variables(results, vars_, metric, args, out_dir):
    a,b=vars_

    # Prefer address as the bar dimension because physical/virtual is the
    # most natural side-by-side comparison in these experiments.
    if b=="address":
        group_param,bar_param=a,b
    elif a=="address":
        group_param,bar_param=b,a
    else:
        group_param,bar_param=a,b

    groups=ordered_values(results,group_param)
    bars=ordered_values(results,bar_param)

    lookup={
        (getattr(r.config,group_param),getattr(r.config,bar_param)):r
        for r in results
    }

    width=0.72/max(1,len(bars))
    x=list(range(len(groups)))

    fig,ax=plt.subplots(figsize=(max(8,1.35*len(groups)),6.0))
    colors=choose_colors([pretty_value(bar_param,v) for v in bars])
    all_bars=[]

    for j,bv in enumerate(bars):
        for i,gv in enumerate(groups):
            r=lookup.get((gv,bv))
            y=metric_value(r,metric) if r else None
            if y is None:
                continue

            pos=i+(j-(len(bars)-1)/2)*width
            bar=ax.bar(
                pos,y,width=width*0.9,
                color=colors[pretty_value(bar_param,bv)],
                label=pretty_value(bar_param,bv) if i==0 else None,
            )[0]
            all_bars.append(bar)

    ax.set_xticks(x)
    ax.set_xticklabels([pretty_value(group_param,v) for v in groups])
    ax.set_xlabel(DISPLAY[group_param])

    style_axis(
        ax,
        f"{metric_label(metric)}: {DISPLAY[bar_param]} comparison",
        fixed_description(args),
        metric_label(metric),
    )

    handles,labels=ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,labels,
            frameon=False,
            ncols=min(3,len(labels)),
            loc="upper center",
            bbox_to_anchor=(0.5,1.18),
        )

    if metric=="accuracy":
        ax.set_ylim(0,100)

    if all_bars:
        fmt=(lambda x:f"{x:.2f}") if metric=="IPC" else (lambda x:f"{x:.1f}%")
        add_value_labels(ax,all_bars,fmt)

    fig.tight_layout(rect=(0,0,1,0.84))
    filename=out_dir/f"{metric_filename(metric)}_{experiment_tag(args)}_vs_{group_param}_by_{bar_param}.png"
    save_fig(fig,filename)


def plot_three_variables(results,vars_,metric,args,out_dir):
    """Plot three varying parameters as a grid of grouped bar charts.

    Special case: when PREFETCHER, CACHE and ADDRESS all vary, CACHE is the
    grid/facet dimension. A no-prefetcher result has CACHE=NONE, which is not
    a real cache placement, so it is NOT given an N/A subplot. For IPC it is
    instead shown as the baseline inside every real-cache subplot. Accuracy
    remains N/A for the no-prefetcher baseline and is therefore omitted.
    """

    # If cache and prefetcher both vary, CACHE=NONE belongs to the
    # no-prefetcher baseline rather than to the normal cache-placement grid.
    # Prefer cores as the facet when it is one of the variables; otherwise
    # cache itself becomes the facet.
    special_baseline=(
        "cache" in vars_ and
        "prefetcher" in vars_ and
        "address" in vars_
    )

    if "cores" in vars_:
        facet="cores"
    elif special_baseline:
        facet="cache"
    else:
        facet=vars_[0]

    remaining=[p for p in vars_ if p!=facet]
    a,b=remaining

    # Keep address as the bar dimension when possible.
    if b=="address":
        group_param,bar_param=a,b
    elif a=="address":
        group_param,bar_param=b,a
    else:
        group_param,bar_param=a,b

    # Real cache placements only. CACHE=NONE belongs to the no-prefetcher
    # baseline and must never become a grid cell.
    facet_vals=ordered_values(results,facet)
    if facet=="cache":
        facet_vals=[v for v in facet_vals if v.upper()!="NONE"]

    if special_baseline:
        active_groups=[
            v for v in ordered_values(results,group_param)
            if v.lower()!="no"
        ]
        baseline_groups=["no"] if any(
            r.config.prefetcher.lower()=="no" and r.config.cache.upper()=="NONE"
            for r in results
        ) and metric=="IPC" else []
        group_vals=active_groups+baseline_groups
    else:
        group_vals=ordered_values(results,group_param)

    bar_vals=ordered_values(results,bar_param)

    n=len(facet_vals)
    ncols=min(3,max(1,math.ceil(math.sqrt(max(1,n)))))
    nrows=max(1,math.ceil(n/max(1,ncols)))

    fig,axes=plt.subplots(
        nrows,ncols,
        figsize=(5.5*ncols,4.25*nrows),
        squeeze=False,
    )
    axes_flat=[ax for row in axes for ax in row]

    colors=choose_colors([pretty_value(bar_param,v) for v in bar_vals])

    lookup={
        (
            getattr(r.config,facet),
            getattr(r.config,group_param),
            getattr(r.config,bar_param),
        ):r
        for r in results
    }

    # Baseline is independent of cache placement.
    baseline_lookup={
        getattr(r.config,bar_param):r
        for r in results
        if r.config.prefetcher.lower()=="no" and r.config.cache.upper()=="NONE"
    }

    width=0.78/max(1,len(bar_vals))

    for idx,fv in enumerate(facet_vals):
        ax=axes_flat[idx]

        subplot_bars=[]
        for j,bv in enumerate(bar_vals):
            for i,gv in enumerate(group_vals):
                r=None

                if special_baseline and gv.lower()=="no":
                    r=baseline_lookup.get(bv)
                else:
                    r=lookup.get((fv,gv,bv))

                y=metric_value(r,metric) if r else None
                if y is None:
                    continue

                pos=i+(j-(len(bar_vals)-1)/2)*width
                bar=ax.bar(
                    pos,y,width=width*0.9,
                    color=colors[pretty_value(bar_param,bv)],
                    label=pretty_value(bar_param,bv) if i==0 else None,
                )[0]
                subplot_bars.append(bar)

        if subplot_bars:
            fmt=(lambda x:f"{x:.2f}") if metric=="IPC" else (lambda x:f"{x:.1f}%")
            add_value_labels(ax,subplot_bars,fmt)

        ax.set_title(
            f"{DISPLAY[facet]} = {pretty_value(facet,fv)}",
            fontsize=11.5,fontweight="bold",loc="left",
        )
        ax.set_xticks(range(len(group_vals)))
        ax.set_xticklabels([pretty_value(group_param,v) for v in group_vals])
        ax.set_xlabel(DISPLAY[group_param],fontsize=9)
        ax.set_ylabel(metric_label(metric),fontsize=9)
        ax.grid(axis="y",linestyle="--",linewidth=0.6,alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.35)
        ax.spines["bottom"].set_alpha(0.35)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(labelsize=8)

        if metric=="accuracy":
            ax.set_ylim(0,100)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Shared legend from the first subplot. Since bars are consistently
    # colored by address (or the selected bar dimension), one legend is enough.
    handles,labels=axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5,0.875),
            ncol=min(4,len(labels)),
        )

    if special_baseline:
        description=(
            f"Grid by {DISPLAY[facet]}; groups by {DISPLAY[group_param]}; "
            f"bars by {DISPLAY[bar_param]}. "
            + ("No-prefetcher IPC is repeated as the baseline in each cache panel. " if metric=="IPC" else "No-prefetcher has no prefetcher accuracy value. ")
            + fixed_description(args)
        )
    else:
        description=(
            f"Grid by {DISPLAY[facet]}; groups by {DISPLAY[group_param]}; "
            f"bars by {DISPLAY[bar_param]}. "
            + fixed_description(args)
        )

    # Explicit header positions avoid the title/subtitle/legend collision that
    # can occur when bbox_inches="tight" is used while saving the figure.
    fig.text(
        0.055,0.965,f"{metric_label(metric)} comparison",
        fontsize=17,fontweight="bold",ha="left",va="top",
    )
    fig.text(
        0.055,0.925,description,
        fontsize=9.5,color="#666666",ha="left",va="top",
    )

    # Leave a clean header area for title, subtitle and shared legend.
    fig.subplots_adjust(
        left=0.07,right=0.98,bottom=0.09,
        top=0.76,hspace=0.34,wspace=0.24,
    )

    filename=out_dir/f"{metric_filename(metric)}_{experiment_tag(args)}_grid_{facet}_{group_param}_by_{bar_param}.png"
    save_fig(fig,filename)

def generate_plots(results, vars_, metric, args, out_dir):
    if not vars_:
        print("ERROR: no variable parameter was specified (use 'none' to vary one).")
        return False

    # A useful special case: when one variable is prefetcher and another is
    # cache, "no" prefetcher rows have cache=NONE. We do not manufacture
    # cache values for them.
    if len(vars_) == 1:
        plot_one_variable(results, vars_[0], metric, args, out_dir)
    elif len(vars_) == 2:
        plot_two_variables(results, vars_, metric, args, out_dir)
    elif len(vars_) == 3:
        plot_three_variables(results, vars_, metric, args, out_dir)
    else:
        raise AssertionError("variable count > 3 should have been rejected")

    return True


def build_parser():
    p = argparse.ArgumentParser(
        description="Generate ChampSim comparison bar plots from result files."
    )
    for name in PARAMS:
        p.add_argument(
            f"--{name}",
            required=True,
            help=f"{DISPLAY[name]} value, or 'none' to make it a variable.",
        )
    p.add_argument(
        "--metric",
        required=True,
        choices=("IPC", "accuracy", "both"),
        help="Metric to plot. 'both' creates separate IPC and accuracy plots.",
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing ChampSim output files (default: results).",
    )
    p.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory where generated plots are written (default: plots).",
    )
    return p



def run_plots(
    trace,
    cache,
    prefetcher,
    address,
    cores,
    warmup,
    sim,
    metric,
    results_dir="results",
    plots_dir="plots",
):
    """Generate plots directly from Python arguments.

    Use lowercase "none" to mean "vary this parameter".
    """

    args=argparse.Namespace(
        trace=trace,
        cache=cache,
        prefetcher=prefetcher,
        address=address,
        cores=cores,
        warmup=warmup,
        sim=sim,
        metric=metric,
        results_dir=results_dir,
        plots_dir=plots_dir,
    )

    errors=validate_semantics(args)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return False

    vars_=variable_params(args)

    if len(vars_)>3:
        print(
            "ERROR: At most 3 variable parameters are allowed. "
            f"Found {len(vars_)}: {', '.join(DISPLAY[p] for p in vars_)}"
        )
        return False

    results_dir=Path(results_dir)
    out_dir=Path(plots_dir)

    if not results_dir.is_dir():
        print(f"ERROR: results directory does not exist: {results_dir}")
        return False

    results,parse_errors=discover_results(results_dir,args)

    if parse_errors:
        print("\nWarnings in matching result files:")
        for e in parse_errors:
            print(f"  - {e}")

    selected,warnings=validate_dataset(results,args)

    if warnings:
        print("\nDataset warnings:")
        for w in warnings:
            print(f"  {w}")

    if not selected:
        print("\nERROR: no usable matching result files found.")
        return False

    if not vars_:
        print(
            "\nERROR: no variable parameter was specified. "
            "Use 'none' for at least one parameter."
        )
        return False

    metrics=("IPC","accuracy") if metric=="both" else (metric,)
    generated=False

    for metric_name in metrics:
        valid=[
            r for r in selected
            if metric_value(r,metric_name) is not None
        ]

        na=[
            r for r in selected
            if metric_value(r,metric_name) is None
        ]

        relevant_na=[
            r for r in na
            if not (metric_name=="accuracy" and r.config.prefetcher.lower()=="no")
        ]

        if relevant_na:
            print(f"\n{metric_name}: {len(relevant_na)} matching result(s) have N/A values:")
            for r in relevant_na:
                print(f"  - {r.path.name}")

        if not valid:
            print(f"\nERROR: no valid data points for metric {metric_name}.")
            continue

        generate_plots(
            valid,
            vars_,
            metric_name,
            args,
            out_dir,
        )
        generated=True

    if generated:
        print(f"\nGenerated plots in: {out_dir.resolve()}")

    return generated


# ============================================================
# CHANGE ONLY THESE ARGUMENTS
# ============================================================

if __name__=="__main__":

    # 1. Vary PREFETCHER and ADDRESS for each fixed trace.
    run_plots(
        trace="600.perlbench_s-210B",
        cache="L2C",
        prefetcher="none",
        address="none",
        cores="1core",
        warmup="100Kwarm",
        sim="1Msim",
        metric="both",
    )

    run_plots(
        trace="602.gcc_s-1850B",
        cache="L2C",
        prefetcher="none",
        address="none",
        cores="1core",
        warmup="100Kwarm",
        sim="1Msim",
        metric="both",
    )

    run_plots(
        trace="605.mcf_s-1152B",
        cache="L2C",
        prefetcher="none",
        address="none",
        cores="1core",
        warmup="100Kwarm",
        sim="1Msim",
        metric="both",
    )

    run_plots(
        trace="619.lbm_s-2676B",
        cache="L2C",
        prefetcher="none",
        address="none",
        cores="1core",
        warmup="100Kwarm",
        sim="1Msim",
        metric="both",
    )

    # 2. Vary TRACE and ADDRESS for each fixed prefetcher.
    run_plots(
        trace="none",
        cache="L2C",
        prefetcher="nextline",
        address="none",
        cores="1core",
        warmup="100Kwarm",
        sim="1Msim",
        metric="both",
    )

    run_plots(
        trace="none",
        cache="L2C",
        prefetcher="ipstride",
        address="none",
        cores="1core",
        warmup="100Kwarm",
        sim="1Msim",
        metric="both",
    )

    run_plots(
        trace="none",
        cache="L2C",
        prefetcher="sppdev",
        address="none",
        cores="1core",
        warmup="100Kwarm",
        sim="1Msim",
        metric="both",
    )

    run_plots(
        trace="none",
        cache="L2C",
        prefetcher="vaampmlite",
        address="none",
        cores="1core",
        warmup="100Kwarm",
        sim="1Msim",
        metric="both",
    )