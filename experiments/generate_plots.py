"""
Comparison figures and tables from the saved run results (results/runs/).

    python experiments/generate_plots.py

Reads the CSV/JSON files written by run_centralized.py, run_federated.py and run_local.py,
and writes to results/:
    summary.md / summary.json      tables used in the README (all numbers computed here)
    global_test_comparison.png     global test MSE and CI per method
    per_client_test.png            each client's own test set, per method
    training_curves.png            test MSE / CI vs training epochs (centralized vs FedAvg)
    local_own_vs_global.png        local-only models on their own vs all clients' drugs
    client_data_distribution.png   training pairs and drugs per client
(each figure also as .pdf)

FedAvg is reported twice:
  - equal budget: the round with the lowest val MSE among rounds 1-50
    (what a 50-round run would have selected; 50 rounds x 2 local epochs = 100 epochs)
  - best: the round selected by the full run (lowest val MSE overall)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RUNS = {"central": "central_full_s42", "fedavg": "fedavg_full_e2_s42", "local": "local_full_s42"}
EQUAL_BUDGET_ROUNDS = 50
NUM_CLIENTS = 5
METRICS = ("mse", "pearson", "ci", "r2")

# reference categorical palette, slots 1-3 (validated all-pairs; aqua needs labels / table view)
COLORS = {"central": "#2a78d6", "fedavg": "#eb6834", "local": "#1baf7a"}
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"central": "Centralized", "fedavg": "FedAvg", "local": "Local-only"}


# ---------------------------------------------------------------- data

def read_run(path):
    out = {"metrics": pd.read_csv(os.path.join(path, "metrics.csv")),
           "clients": pd.read_csv(os.path.join(path, "client_metrics.csv"))}
    with open(os.path.join(path, "final_metrics.json")) as f:
        out["final"] = json.load(f)
    return out


def config(path):
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


def client_test(clients, step_key, step):
    """Per-client test metrics {k: {mse, pearson, ci, r2}} at one epoch/round from client_metrics.csv."""
    rows = clients[(clients[step_key] == step) & (clients["split"] == "test")]
    return {int(r["client"]): {m: float(r[m]) for m in METRICS}
            for _, r in rows.iterrows() if r["client"] != "all"}


def gap_closed(local, fed, central):
    """(local - FedAvg) / (local - centralized): share of the local-to-centralized gap closed by FedAvg."""
    return (local - fed) / (local - central)


def collect(runs_dir):
    central = read_run(os.path.join(runs_dir, RUNS["central"]))
    fedavg = read_run(os.path.join(runs_dir, RUNS["fedavg"]))
    local_dir = os.path.join(runs_dir, RUNS["local"])
    local = [read_run(os.path.join(local_dir, f"client_{k}")) for k in range(NUM_CLIENTS)]
    cfg = {"central": config(os.path.join(runs_dir, RUNS["central"])),
           "fedavg": config(os.path.join(runs_dir, RUNS["fedavg"])),
           "local": config(local_dir)}

    s = {"methods": {}, "per_client": {}, "budget": {}, "gap_closed": {}}

    # centralized: best epoch
    cf = central["final"]
    s["methods"]["central"] = {m: cf["test"]["all"][m] for m in METRICS}
    s["per_client"]["central"] = {k: {m: cf["test"][str(k)][m] for m in METRICS} for k in range(NUM_CLIENTS)}
    s["budget"]["central"] = {"max_epochs": cfg["central"]["epochs"], "patience": cfg["central"]["patience"],
                              "epochs_run": int(central["metrics"]["epoch"].max()),
                              "best_epoch": cf["best_epoch"], "stopped_early": cf["stopped_early"]}

    # FedAvg best: round selected by the full run
    ff, fm = fedavg["final"], fedavg["metrics"]
    s["methods"]["fedavg_best"] = {m: ff["test"]["all"][m] for m in METRICS}
    s["per_client"]["fedavg_best"] = {k: {m: ff["test"][str(k)][m] for m in METRICS} for k in range(NUM_CLIENTS)}

    # FedAvg equal budget: lowest val MSE among rounds 1-50
    first = fm[fm["round"] <= EQUAL_BUDGET_ROUNDS]
    eq_round = int(first.loc[first["val_mse"].idxmin(), "round"])
    row = fm[fm["round"] == eq_round].iloc[0]
    s["methods"]["fedavg_equal"] = {m: float(row[f"test_{m}"]) for m in METRICS}
    s["per_client"]["fedavg_equal"] = client_test(fedavg["clients"], "round", eq_round)

    le = cfg["fedavg"]["local_epochs"]
    s["budget"]["fedavg"] = {"max_rounds": cfg["fedavg"]["rounds"], "local_epochs": le,
                             "patience": cfg["fedavg"]["patience"], "rounds_run": int(fm["round"].max()),
                             "best_round": ff["best_round"], "stopped_early": ff["stopped_early"],
                             "equal_budget_rounds": EQUAL_BUDGET_ROUNDS, "equal_budget_selected_round": eq_round}

    # local-only: each client's best epoch; own test and global test
    s["methods"]["local"] = {m: float(np.mean([l["final"]["test"]["all"][m] for l in local])) for m in METRICS}
    s["per_client"]["local_own"] = {k: {m: local[k]["final"]["test"][str(k)][m] for m in METRICS}
                                    for k in range(NUM_CLIENTS)}
    s["per_client"]["local_global"] = {k: {m: local[k]["final"]["test"]["all"][m] for m in METRICS}
                                       for k in range(NUM_CLIENTS)}
    s["budget"]["local"] = {"max_epochs": cfg["local"]["epochs"], "patience": cfg["local"]["patience"],
                            "per_client": {k: {"epochs_run": int(local[k]["metrics"]["epoch"].max()),
                                               "best_epoch": local[k]["final"]["best_epoch"],
                                               "stopped_early": local[k]["final"]["stopped_early"],
                                               "n_train": None}
                                           for k in range(NUM_CLIENTS)}}
    summ = pd.read_csv(os.path.join(local_dir, "local_summary.csv"))
    for _, r in summ.iterrows():
        s["budget"]["local"]["per_client"][int(r["client"])]["n_train"] = int(r["n_train"])

    # gap closed, global test
    L, C = s["methods"]["local"], s["methods"]["central"]
    for key in ("fedavg_equal", "fedavg_best"):
        F = s["methods"][key]
        s["gap_closed"][key] = {m: gap_closed(L[m], F[m], C[m]) for m in ("mse", "ci")}

    return s, central, fedavg, local


# ---------------------------------------------------------------- tables

def markdown(s):
    b = s["budget"]
    fb = b["fedavg"]
    lines = []
    rows = [
        ("Centralized", "central",
         f"{b['central']['epochs_run']} epochs (best {b['central']['best_epoch']})"),
        (f"FedAvg, equal budget (best of rounds 1-{fb['equal_budget_rounds']})", "fedavg_equal",
         f"{fb['equal_budget_rounds']} rounds x {fb['local_epochs']} local epochs "
         f"(selected round {fb['equal_budget_selected_round']})"),
        ("FedAvg, best", "fedavg_best",
         f"{fb['rounds_run']} rounds x {fb['local_epochs']} local epochs (best round {fb['best_round']})"),
        ("Local-only (mean of 5 clients)", "local", "up to 100 epochs per client"),
    ]
    lines.append("| Method | Training budget | MSE | Pearson | CI | R2 |")
    lines.append("|---|---|---|---|---|---|")
    for name, key, budget in rows:
        m = s["methods"][key]
        lines.append(f"| {name} | {budget} | {m['mse']:.3f} | {m['pearson']:.3f} | {m['ci']:.3f} | {m['r2']:.2f} |")

    lines.append("")
    lines.append("| FedAvg | Gap closed, MSE | Gap closed, CI |")
    lines.append("|---|---|---|")
    for name, key in (("Equal budget", "fedavg_equal"), ("Best", "fedavg_best")):
        g = s["gap_closed"][key]
        lines.append(f"| {name} | {100 * g['mse']:.0f}% | {100 * g['ci']:.0f}% |")

    lines.append("")
    lines.append("| Client | Train pairs | Local-only (own) | FedAvg, equal budget | FedAvg, best | Centralized |")
    lines.append("|---|---|---|---|---|---|")
    pc = s["per_client"]
    for k in range(NUM_CLIENTS):
        cells = [f"{pc[key][k]['mse']:.3f} / {pc[key][k]['ci']:.3f}"
                 for key in ("local_own", "fedavg_equal", "fedavg_best", "central")]
        lines.append(f"| {k} | {b['local']['per_client'][k]['n_train']:,} | " + " | ".join(cells) + " |")
    means = [f"{np.mean([pc[key][k]['mse'] for k in range(NUM_CLIENTS)]):.3f} / "
             f"{np.mean([pc[key][k]['ci'] for k in range(NUM_CLIENTS)]):.3f}"
             for key in ("local_own", "fedavg_equal", "fedavg_best", "central")]
    lines.append("| Mean | | " + " | ".join(means) + " |")

    lines.append("")
    lines.append("| Local-only model | Best epoch | Epochs run | Own test MSE / CI | Global test MSE / CI |")
    lines.append("|---|---|---|---|---|")
    for k in range(NUM_CLIENTS):
        pk = b["local"]["per_client"][k]
        own, glob = pc["local_own"][k], pc["local_global"][k]
        lines.append(f"| Client {k} | {pk['best_epoch']} | {pk['epochs_run']}"
                     f"{'' if pk['stopped_early'] else ' (limit)'} | {own['mse']:.3f} / {own['ci']:.3f} | "
                     f"{glob['mse']:.3f} / {glob['ci']:.3f} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- figures

def style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)


def save(fig, out_dir, name, bottom=0.0):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved results/{name}.png")


def fig_global(s, out_dir):
    keys = ["local", "fedavg_equal", "fedavg_best", "central"]
    names = ["Local-only\n(mean)", "FedAvg\nequal budget", "FedAvg\nbest", "Centralized"]
    colors = [COLORS["local"], COLORS["fedavg"], COLORS["fedavg"], COLORS["central"]]
    hatches = ["", "///", "", ""]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, label in zip(axes, ("mse", "ci"), ("Test MSE (lower is better)", "Test CI (higher is better)")):
        vals = [s["methods"][k][metric] for k in keys]
        bars = ax.bar(names, vals, color=colors, hatch=hatches, edgecolor=SURFACE, linewidth=2, width=0.62)
        for bar in bars[1:2]:
            bar.set_edgecolor(COLORS["fedavg"])
            bar.set_facecolor("none")
            bar.set_linewidth(1.5)
        # individual local-only clients as dots on the local bar
        indiv = [s["per_client"]["local_global"][k][metric] for k in range(NUM_CLIENTS)]
        ax.scatter([0] * len(indiv), indiv, s=22, color=INK_MUTED, zorder=3, label="individual local clients")
        for i, (bar, v) in enumerate(zip(bars, vals)):
            top = max(v, max(indiv)) if i == 0 else v  # local-only: label above its client dots
            ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, top), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
        style(ax, label)
        ax.set_ylim(*((0.5, 0.93) if metric == "ci" else (0, 1.0)))
        ax.legend(frameon=False, fontsize=8, loc="upper right" if metric == "mse" else "upper left",
                  labelcolor=INK_MUTED)
    fig.suptitle("Global test set (6,011 pairs, all clients' drugs)", color=INK, fontsize=11)
    save(fig, out_dir, "global_test_comparison")


def fig_per_client(s, out_dir):
    keys = [("local_own", "Local-only (own model)", COLORS["local"]),
            ("fedavg_best", "FedAvg (best)", COLORS["fedavg"]),
            ("central", "Centralized", COLORS["central"])]
    x = np.arange(NUM_CLIENTS)
    w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, label in zip(axes, ("mse", "ci"), ("Own test MSE (lower is better)", "Own test CI (higher is better)")):
        for i, (key, name, color) in enumerate(keys):
            vals = [s["per_client"][key][k][metric] for k in range(NUM_CLIENTS)]
            ax.bar(x + (i - 1) * w, vals, w, color=color, edgecolor=SURFACE, linewidth=2, label=name)
        ax.set_xticks(x, [f"Client {k}" for k in range(NUM_CLIENTS)])
        style(ax, label)
        ax.set_ylim(*((0.7, 0.93) if metric == "ci" else (0, 0.5)))  # headroom for the legend
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=3, loc="upper left")
    fig.suptitle("Each client's own test pairs (values in the README table)", color=INK, fontsize=11)
    save(fig, out_dir, "per_client_test")


def fig_curves(s, central, fedavg, out_dir):
    cm, fm = central["metrics"], fedavg["metrics"]
    le = s["budget"]["fedavg"]["local_epochs"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, label in zip(axes, ("mse", "ci"), ("Test MSE", "Test CI")):
        ax.plot(cm["epoch"], cm[f"test_{metric}"], color=COLORS["central"], linewidth=2, label="Centralized")
        ax.plot(fm["round"] * le, fm[f"test_{metric}"], color=COLORS["fedavg"], linewidth=2,
                label=f"FedAvg (1 round = {le} local epochs)")
        ax.axhline(s["methods"]["local"][metric], color=COLORS["local"], linewidth=2, linestyle="--",
                   label="Local-only (mean, global test)")
        # selected models
        b = s["budget"]
        sel = [(b["central"]["best_epoch"], s["methods"]["central"][metric], COLORS["central"]),
               (b["fedavg"]["equal_budget_selected_round"] * le, s["methods"]["fedavg_equal"][metric], COLORS["fedavg"]),
               (b["fedavg"]["best_round"] * le, s["methods"]["fedavg_best"][metric], COLORS["fedavg"])]
        for xv, yv, c in sel:
            ax.scatter([xv], [yv], s=64, color=c, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.axvline(b["fedavg"]["equal_budget_rounds"] * le, color=INK_MUTED, linewidth=1, linestyle=":")
        ax.annotate("equal budget (100 epochs)", (b["fedavg"]["equal_budget_rounds"] * le, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points",
                    fontsize=8, color=INK_MUTED, ha="center", va="bottom")
        ax.set_xlabel("Training epochs (FedAvg: rounds x local epochs)", color=INK_MUTED, fontsize=10)
        style(ax, label)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK_MUTED)
    fig.suptitle("Test metrics during training (dots: models selected on validation MSE)", color=INK, fontsize=11)
    save(fig, out_dir, "training_curves", bottom=0.08)


def fig_local_gap(s, out_dir):
    x = np.arange(NUM_CLIENTS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, label in zip(axes, ("mse", "ci"), ("MSE (lower is better)", "CI (higher is better)")):
        own = [s["per_client"]["local_own"][k][metric] for k in range(NUM_CLIENTS)]
        glob = [s["per_client"]["local_global"][k][metric] for k in range(NUM_CLIENTS)]
        ax.bar(x - 0.18, own, 0.36, color=COLORS["local"], edgecolor=SURFACE, linewidth=2, label="Own test pairs")
        ax.bar(x + 0.18, glob, 0.36, facecolor="none", edgecolor=COLORS["local"], hatch="///", linewidth=1.5,
               label="Global test pairs (mostly other clients' drugs)")
        ax.set_xticks(x, [f"Client {k}" for k in range(NUM_CLIENTS)])
        style(ax, label)
        ax.set_ylim(*((0.5, 0.97) if metric == "ci" else (0, 1.12)))  # headroom for the legend
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left")
    fig.suptitle("Local-only models: own drugs vs all clients' drugs", color=INK, fontsize=11)
    save(fig, out_dir, "local_own_vs_global")


def fig_clients(out_dir):
    from utils.protocol import load_protocol
    proto = load_protocol()
    x = np.arange(len(proto["clients"]))
    n_train = [len(d["train"]) for d in proto["clients"]]
    drugs = [pd.concat([d[s] for s in ("train", "val", "test")])["Drug"].nunique() for d in proto["clients"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, vals, label, fmt in ((axes[0], n_train, "Training pairs", "{:,}"), (axes[1], drugs, "Unique drugs", "{}")):
        # neutral color: this chart is about the data, not a method
        bars = ax.bar(x, vals, color=INK_MUTED, edgecolor=SURFACE, linewidth=2, width=0.6)
        for bar, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.set_xticks(x, [f"Client {k}" for k in x])
        style(ax, label)
    fig.suptitle("Drug-based non-IID partition (each client holds different drugs)", color=INK, fontsize=11)
    save(fig, out_dir, "client_data_distribution")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=os.path.join(ROOT, "results", "runs"))
    parser.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = parser.parse_args()

    s, central, fedavg, local = collect(args.runs)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    md = markdown(s)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)

    fig_global(s, args.out)
    fig_per_client(s, args.out)
    fig_curves(s, central, fedavg, args.out)
    fig_local_gap(s, args.out)
    fig_clients(args.out)


if __name__ == "__main__":
    main()
