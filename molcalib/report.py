"""Report (T10): results/summary.md and figures, generated only from the results files.

No numbers are typed by hand: every number in summary.md comes from results/*.csv or
results/split_audit.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from molcalib import config
from molcalib.conformal import VARIANTS

# Reference palette (dataviz skill), slots 1-2, validated on the light surface.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SOURCE_COLOR = {"naive_cal": "#2a78d6", "valid": "#eb6834"}
METHOD_COLOR = {"ensemble": "#2a78d6", "mcd": "#eb6834"}
METHOD_NAME = {"ensemble": "Deep ensemble", "mcd": "MC dropout"}
TASK_NAME = {"caco2_wang": "Caco-2", "lipophilicity_astrazeneca": "Lipophilicity",
             "solubility_aqsoldb": "Solubility (AqSolDB)"}
TARGET = 1 - config.ALPHA


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)


def _target_line(ax):
    ax.axhline(TARGET, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)


def _fig(nrows, ncols, w=4.2, h=3.2):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def fig_coverage_by_source(summary: pd.DataFrame, out: Path) -> None:
    """Coverage per conformal variant, naive_cal vs valid, with calibration-resampled 95% CIs."""
    cov = summary[summary["metric"] == "coverage"]
    fig, axes = _fig(2, len(config.TASKS))
    for r, method in enumerate(("ensemble", "mcd")):
        for c, task in enumerate(config.TASKS):
            ax = axes[r, c]
            _style(ax, f"{TASK_NAME[task]} · {METHOD_NAME[method]}")
            _target_line(ax)
            for k, source in enumerate(("naive_cal", "valid")):
                d = cov[(cov.task == task) & (cov.method == method) & (cov.source == source)]
                d = d.set_index("variant").loc[list(VARIANTS)]
                x = [i + (k - 0.5) * 0.28 for i in range(len(VARIANTS))]
                ax.errorbar(x, d["mean"], yerr=[d["mean"] - d.ci_lo_cal, d.ci_hi_cal - d["mean"]],
                            fmt="o", ms=6, color=SOURCE_COLOR[source], ecolor=SOURCE_COLOR[source],
                            elinewidth=2, capsize=0, label=source, zorder=3)
            ax.set_xticks(range(len(VARIANTS)))
            ax.set_xticklabels([v.replace("_", "\n") for v in VARIANTS])
            ax.set_ylim(0.7, 1.0)
            if c == 0:
                ax.set_ylabel("test coverage", fontsize=9, color=INK_2)
    axes[0, 0].legend(title="calibration set", fontsize=8, title_fontsize=8, frameon=False,
                      loc="lower left")
    fig.suptitle(f"Conformal coverage by calibration source (target {TARGET:.0%}, dashed); "
                 "bars: 95% CI with calibration resampling", fontsize=10, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_coverage_by_nn_bin(bins: pd.DataFrame, out: Path) -> None:
    """Coverage per test distance quartile (mean over seeds), deep ensemble."""
    d = bins[bins.method == "ensemble"].groupby(["task", "source", "variant", "bin"])["coverage"].mean()
    fig, axes = _fig(1, len(config.TASKS))
    for c, task in enumerate(config.TASKS):
        ax = axes[0, c]
        _style(ax, TASK_NAME[task])
        _target_line(ax)
        for source in ("naive_cal", "valid"):
            for variant, ls in (("split", "-"), ("mondrian", (0, (2, 2)))):
                s = d.loc[(task, source, variant)]
                ax.plot(s.index + 1, s.values, ls=ls, lw=2, marker="o", ms=6,
                        color=SOURCE_COLOR[source], label=f"{variant}, {source}")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("test distance-to-train quartile (1 = nearest)", fontsize=8, color=INK_2)
        ax.set_ylim(0.6, 1.0)
    axes[0, 0].set_ylabel("test coverage", fontsize=9, color=INK_2)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Coverage by distance to the training set (deep ensemble, mean of 5 seeds)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_selective(sel: pd.DataFrame, out: Path) -> None:
    """RMSE when keeping only the most confident fraction (mean over seeds, band = ±1 sd)."""
    g = sel.groupby(["task", "method", "keep_frac"])["rmse"].agg(["mean", "std"])
    fig, axes = _fig(1, len(config.TASKS))
    for c, task in enumerate(config.TASKS):
        ax = axes[0, c]
        _style(ax, TASK_NAME[task])
        for method in ("ensemble", "mcd"):
            s = g.loc[(task, method)].sort_index(ascending=False)
            ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"],
                            color=METHOD_COLOR[method], alpha=0.12, linewidth=0)
            ax.plot(s.index, s["mean"], lw=2, marker="o", ms=6, color=METHOD_COLOR[method],
                    label=METHOD_NAME[method])
        ax.invert_xaxis()
        ax.set_xlabel("fraction kept (most confident first)", fontsize=8, color=INK_2)
    axes[0, 0].set_ylabel("test RMSE", fontsize=9, color=INK_2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Selective prediction: error on the most confident molecules "
                 "(mean of 5 seeds, band ±1 sd)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- tables


def _f(x, nd=3, sign=False):
    if pd.isna(x):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _ci(lo, hi, nd=3, sign=False):
    if pd.isna(lo) and pd.isna(hi):
        return "—"  # not applicable (e.g. no conformal step, so nothing to recalibrate)
    return f"[{_f(lo, nd, sign)}, {_f(hi, nd, sign)}]"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def table_point(summary: pd.DataFrame) -> str:
    names = {"mae": "MAE", "rmse": "RMSE", "nll": "NLL", "miscal_area": "Miscal. area",
             "miscal_signed": "Miscal. signed", "spearman_sigma_abserr": "Spearman σ vs abs. error"}
    rows = []
    for task in config.TASKS:
        for method in ("ensemble", "mcd"):
            d = summary[(summary.task == task) & (summary.method == method)
                        & summary.source.isna()].set_index("metric")
            rows.append({"Task": TASK_NAME[task], "Method": METHOD_NAME[method],
                         **{n: f"{_f(d.at[m, 'mean'])} ± {_f(d.at[m, 'sd'])} "
                               f"{_ci(d.at[m, 'ci_lo'], d.at[m, 'ci_hi'])}"
                            for m, n in names.items()}})
    return _md(pd.DataFrame(rows))


def table_conformal(summary: pd.DataFrame, metric: str) -> str:
    rows = []
    d = summary[summary.metric == metric]
    for task in config.TASKS:
        for method in ("ensemble", "mcd"):
            for variant in VARIANTS:
                row = {"Task": TASK_NAME[task], "Method": METHOD_NAME[method], "Variant": variant}
                for source in ("naive_cal", "valid"):
                    r = d[(d.task == task) & (d.method == method) & (d.variant == variant)
                          & (d.source == source)].iloc[0]
                    row[source] = (f"{_f(r['mean'])} {_ci(r.ci_lo, r.ci_hi)} "
                                   f"{_ci(r.ci_lo_cal, r.ci_hi_cal)}")
                rows.append(row)
    return _md(pd.DataFrame(rows))


def table_paired(paired: pd.DataFrame, comparison: str, metrics: list[str]) -> str:
    d = paired[(paired.comparison == comparison) & paired.metric.isin(metrics)].copy()
    if comparison == "ensemble-mcd":
        d = d[d.source.isna() | (d.variant == "split")]
    rows = []
    for r in d.itertuples(index=False):
        rows.append({"Task": TASK_NAME[r.task],
                     "Method": METHOD_NAME.get(r.method, "") if isinstance(r.method, str) else "",
                     "Variant": r.variant if isinstance(r.variant, str) else "",
                     "Source": r.source if isinstance(r.source, str) else "",
                     "Metric": r.metric, "Mean diff": _f(r.mean_diff, sign=True),
                     "95% CI (test)": _ci(r.ci_lo, r.ci_hi, sign=True),
                     "95% CI (test+cal)": _ci(r.ci_lo_cal, r.ci_hi_cal, sign=True),
                     "Seeds > 0": f"{r.n_seeds_positive}/{r.n_seeds}"})
    df = pd.DataFrame(rows)
    return _md(df.loc[:, (df != "").any()])


def table_selective(sel: pd.DataFrame) -> str:
    g = sel.groupby(["task", "method", "keep_frac"])["rmse"].mean().unstack("keep_frac")
    g = g[sorted(g.columns, reverse=True)]
    rows = [{"Task": TASK_NAME[t], "Method": METHOD_NAME[m],
             **{f"keep {k:.0%}": _f(v) for k, v in r.items()}} for (t, m), r in g.iterrows()]
    return _md(pd.DataFrame(rows))


def table_compute(results: Path, summary: pd.DataFrame) -> str:
    comp = pd.concat([pd.read_csv(f) for f in sorted(results.glob("compute_full_*.csv"))])
    comp = comp[comp.status == "ok"]
    rows = []
    for task in config.TASKS:
        for method in ("ensemble", "mcd"):
            c = comp[(comp.task == task) & (comp.method == method)]
            s = summary[(summary.task == task) & (summary.method == method)]
            nll = s[s.metric == "nll"].iloc[0]
            cov = s[(s.metric == "coverage") & (s.variant == "split") & (s.source == "valid")].iloc[0]
            rows.append({"Task": TASK_NAME[task], "Method": METHOD_NAME[method],
                         "Models per seed": int(c.n_models.iloc[0]),
                         "GPU-min, 5 seeds": _f(c.train_s.sum() / 60, 1),
                         "GPU-min per model": _f(c.train_s.sum() / c.n_models.sum() / 60, 2),
                         "NLL": _f(nll["mean"]),
                         "Split coverage (valid cal)": _f(cov["mean"])})
    return _md(pd.DataFrame(rows))


def table_audit(audit: dict) -> str:
    rows = []
    for task in config.TASKS:
        runs = [v for k, v in audit["runs"].items() if k.startswith(task)]
        n = runs[0]["n"]
        rows.append({"Task": TASK_NAME[task],
                     **{s: n.get(s, 0) for s in ("train_fit", "naive_cal", "valid", "test")},
                     "Max scaffold overlap (5 seeds)": max(max(r["scaffold_overlap"].values())
                                                          for r in runs),
                     "naive_cal sharing a train scaffold (mean)": _f(sum(
                         r["naive_cal_frac_molecules_sharing_scaffold_with_train_fit"]
                         for r in runs) / len(runs), 2)})
    return _md(pd.DataFrame(rows))


def write_report(results: Path) -> Path:
    summary = pd.read_csv(results / "stats_summary.csv")
    paired = pd.read_csv(results / "stats_paired.csv")
    sel = pd.read_csv(results / "metrics_selective.csv")
    bins = pd.read_csv(results / "metrics_nn_bins.csv")
    audit = json.loads((results / "split_audit.json").read_text())
    commit = summary["git_commit"].iloc[0]

    fig_coverage_by_source(summary, results / "fig_coverage_by_source.png")
    fig_coverage_by_nn_bin(bins, results / "fig_coverage_by_nn_bin.png")
    fig_selective(sel, results / "fig_selective.png")

    seeds = ", ".join(map(str, config.SEEDS))
    text = f"""# MolCalib results summary

Generated by `scripts/evaluate.py` from the files in `results/`; no number below is typed by hand.
Code commit of the statistics: `{commit}`.

Setup: tasks {", ".join(f"`{t}`" for t in config.TASKS)}; split seeds {seeds};
target coverage {TARGET:.0%} (alpha = {config.ALPHA}); {config.N_MEMBERS}-member deep ensemble and
MC dropout with T = {config.MC_T}; {config.N_BOOT} scaffold-group bootstrap resamples.

How to read the intervals:
- "mean ± sd" is over the {len(config.SEEDS)} split seeds on the full test set.
- "[a, b]" after a value is a 95% CI from resampling test scaffold groups only (models and
  calibration fixed).
- A second "[a, b]" (conformal tables) also resamples calibration scaffold groups and refits the
  conformal quantiles (DR-005). Use this one for questions about the calibration source.
- No correction for multiple comparisons is applied.

## 1. Splits

{table_audit(audit)}

Split sizes are for seed {config.SEEDS[0]}. ![distance to training set](fig_nn_distance.png)

## 2. Accuracy and model uncertainty (test set)

Miscal. signed < 0 means the model's own Gaussian intervals are too narrow.

{table_point(summary)}

## 3. Conformal coverage (target {TARGET:.0%})

Each cell: mean, 95% CI (test), 95% CI (test + calibration).

{table_conformal(summary, "coverage")}

![coverage by calibration source](fig_coverage_by_source.png)

### Mean interval width

{table_conformal(summary, "mean_width")}

### Coverage in the farthest test distance quartile

{table_conformal(summary, "coverage_far_bin")}

![coverage by distance](fig_coverage_by_nn_bin.png)

## 4. Paired comparisons

### valid minus naive_cal calibration

{table_paired(paired, "valid-naive_cal", ["coverage", "coverage_far_bin", "mean_width"])}

### Deep ensemble minus MC dropout

Point metrics, and split-conformal coverage and width. "—": not applicable (point metrics have
no calibration step to resample).

{table_paired(paired, "ensemble-mcd", ["mae", "rmse", "nll", "miscal_area",
                                         "spearman_sigma_abserr", "coverage", "mean_width"])}

## 5. Selective prediction (test RMSE, mean of seeds)

{table_selective(sel)}

![selective prediction](fig_selective.png)

## 6. Compute cost (Kaggle, 2 × T4)

{table_compute(results, summary)}
"""
    path = results / "summary.md"
    path.write_text(text)
    return path
