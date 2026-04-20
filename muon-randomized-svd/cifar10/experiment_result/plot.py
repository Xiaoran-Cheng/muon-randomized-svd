import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_curve(leaf_dir, metric="val_loss"):
    """metric in {'train_loss','train_acc','val_loss','val_acc'}. Returns (steps, mean, std)."""
    data = pd.read_pickle(os.path.join(leaf_dir, "validation_metrics.pkl"))
    df = pd.DataFrame(data["values"].numpy(), index=data["eval_steps"], columns=data["columns"])
    cols = df.filter(like=metric)
    return df.index.to_numpy(), cols.mean(axis=1).to_numpy(), cols.std(axis=1).to_numpy()


def load_test_acc(leaf_dir):
    """Return (mean, std) of final test accuracy across trials."""
    data = pd.read_pickle(os.path.join(leaf_dir, "accuracy.pkl"))
    df = pd.DataFrame(data["values"].numpy(), index=data["rows"], columns=data["columns"])
    test = df["test_acc_tta2"]
    return float(test.mean()), float(test.std())


def plot_group(group_dir, labels, title, out_png, metric="val_loss"):
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for name, label in labels.items():
        steps, mean, std = load_curve(os.path.join(group_dir, name), metric)
        ax.plot(steps, mean, label=label)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("step", fontsize=18)
    ax.set_ylabel(metric.replace("_", " "), fontsize=18)
    ax.tick_params(axis="both", labelsize=18)
    # ax.set_title(title)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"saved {out_png}")


def acc_table(group_dir, labels, caption, group_name):
    rows = []
    for name, label in labels.items():
        mean, std = load_test_acc(os.path.join(group_dir, name))
        rows.append({"group": group_name, "method": label,
                     "test_acc_mean": mean, "test_acc_std": std})
    table = pd.DataFrame(rows)
    print(f"\n=== {caption} ===")
    print(table.set_index("method")[["test_acc_mean", "test_acc_std"]]
          .to_string(float_format=lambda x: f"{x:.4f}"))
    return table


all_tables = []


# ---------------- what to plot ----------------
# any subset of {"train_loss", "train_acc", "val_loss", "val_acc"}
METRICS = ["val_loss", "train_loss"]


# ---------------- E5 methods comparison ----------------
E5_DIR = os.path.join(RESULT_DIR, "E5_methods_comparison")
E5_LABELS = {
    "AdamW":             "AdamW",
    "SGDNesterov":       "SGD-Nesterov",
    "MuonPolyakFull":    "Muon Polyak (full SVD)",
    "MuonPolyakRand":    "Muon Polyak (rand SVD)",
    "MuonNesterovFull":  "Muon Nesterov (full SVD)",
    "MuonNesterovRand":  "Muon Nesterov (rand SVD)",
}
for m in METRICS:
    plot_group(E5_DIR, E5_LABELS, f"E5: {m.replace('_', ' ')}",
               os.path.join(RESULT_DIR, f"E5_{m}.png"), metric=m)
all_tables.append(acc_table(E5_DIR, E5_LABELS, "E5: final test accuracy (mean ± std)", "E5"))


# ---------------- Ablations ----------------
ABL_DIR = os.path.join(RESULT_DIR, "Ablation")
ABLATIONS = {
    "E1_solvers": {
        "polar_express":       "Polar Express",
        "cubic_theoretical":   "Cubic (theoretical)",
        "quintic_theoretical": "Quintic (theoretical)",
        "quintic_empirical":   "Quintic (empirical)",
    },
    "E2_rank": {f"rank_{r}": f"rank = {r}" for r in [16, 32, 64, 128, 256]},
    "E3_NS_step": {f"q_{q}": f"q = {q}" for q in [1, 3, 5, 7, 9]},
    "E4_batch_size": {f"bs_{b}": f"bs = {b}" for b in [500, 1000, 2000, 3000, 4000]},
}
for sub, labels in ABLATIONS.items():
    for m in METRICS:
        plot_group(os.path.join(ABL_DIR, sub), labels,
                   f"Ablation {sub}: {m.replace('_', ' ')}",
                   os.path.join(RESULT_DIR, f"Ablation_{sub}_{m}.png"), metric=m)
    all_tables.append(acc_table(os.path.join(ABL_DIR, sub), labels,
                                f"Ablation {sub}: final test accuracy (mean ± std)",
                                f"Ablation_{sub}"))

csv_path = os.path.join(RESULT_DIR, "test_accuracy_tables.csv")
pd.concat(all_tables, ignore_index=True).to_csv(csv_path, index=False)
print(f"\nsaved {csv_path}")
