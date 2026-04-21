"""
Aggregate 6 methods' training logs on the server.
For each method: average across trials, then plot train/val loss/acc
(one figure per metric, 6 lines), and dump an averaged accuracy table.
"""
import os
import glob
import pickle
import pandas as pd
import matplotlib.pyplot as plt

LOGS_ROOT = "/home/xcheng328/cifar10/logs"
MUON_RAND_DIR = (
    "/home/xcheng328/cifar10/final_MuonNesterovRand/logs/"
    "ommuon_bs1000_mlr0p195723_mmm0p483562_mnt_iscubic_ns_theoretical_"
    "os9_rzt_rk256_ov10_pi2"
)

METHODS = {
    "adamw":              os.path.join(LOGS_ROOT, "adamw"),
    "sgd-nesterov":       os.path.join(LOGS_ROOT, "sgd-nesterov"),
    "muon-nesterov-full": os.path.join(LOGS_ROOT, "muon-nesterov-full"),
    "muon-polyak-full":   os.path.join(LOGS_ROOT, "muon-polyak-full"),
    "muon-polyak-rand":   os.path.join(LOGS_ROOT, "muon-polyak-rand"),
    "muon-nesterov-rand": MUON_RAND_DIR,
}

METRICS = ["train_loss", "train_acc", "val_loss", "val_acc"]


def resolve_run_dir(path: str) -> str:
    """If `path` already contains the pkl files, return it; else pick its first subdir."""
    if os.path.exists(os.path.join(path, "validation_metrics.pkl")):
        return path
    subs = [d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d)
            and os.path.exists(os.path.join(d, "validation_metrics.pkl"))]
    if not subs:
        raise FileNotFoundError(f"no run dir with validation_metrics.pkl under {path}")
    if len(subs) > 1:
        print(f"[warn] multiple runs under {path}, using {subs[0]}")
    return subs[0]


def load_pkl(p: str):
    with open(p, "rb") as f:
        return pickle.load(f)


def to_dataframe(d: dict) -> pd.DataFrame:
    """dict with eval_steps / columns / values -> DataFrame indexed by step."""
    vals = d["values"]
    if hasattr(vals, "cpu"):
        vals = vals.cpu().numpy()
    df = pd.DataFrame(vals, index=d["eval_steps"], columns=d["columns"])
    df.index.name = "step"
    return df


def avg_by_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Columns containing a metric keyword (e.g. 'train_loss') are averaged together."""
    out = {}
    for m in METRICS:
        cols = [c for c in df.columns if m in c.lower()]
        if cols:
            out[m] = df[cols].mean(axis=1)
    return pd.DataFrame(out, index=df.index)


def summarize_acc(acc) -> pd.Series:
    """accuracy.pkl = {rows: [trial_ids], columns: [metric_names], values: (n_trials, n_metrics)}.
       Average across rows (trials) -> one value per metric."""
    vals = acc["values"]
    if hasattr(vals, "cpu"):
        vals = vals.cpu().numpy()
    df = pd.DataFrame(vals, index=acc.get("rows"), columns=acc["columns"])
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    mean.index = [f"{c}_mean" for c in mean.index]
    std.index = [f"{c.replace('_mean','')}_std" for c in mean.index]
    return pd.concat([mean, std])


def load_method(name: str, path: str):
    run_dir = resolve_run_dir(path)
    print(f"[{name}] {run_dir}")

    val_avg = None
    try:
        val = to_dataframe(load_pkl(os.path.join(run_dir, "validation_metrics.pkl")))
        val_avg = avg_by_metric(val)
        print(f"  val_metrics: {val.shape} -> {val_avg.shape}, metrics={list(val_avg.columns)}")
    except Exception as e:
        print(f"  [val] FAILED: {e}")

    acc_series = None
    acc_path = os.path.join(run_dir, "accuracy.pkl")
    if os.path.exists(acc_path):
        try:
            acc = load_pkl(acc_path)
            print(f"  accuracy.pkl type={type(acc).__name__}", end="")
            if isinstance(acc, dict):
                print(f", keys={list(acc.keys())[:10]}")
            else:
                print()
            acc_series = summarize_acc(acc)
        except Exception as e:
            print(f"  [acc] FAILED: {e}")
    return val_avg, acc_series


def main():
    per_method_val, per_method_acc = {}, {}
    for name, path in METHODS.items():
        try:
            v, a = load_method(name, path)
            if v is not None:
                per_method_val[name] = v
            if a is not None:
                per_method_acc[name] = a
        except Exception as e:
            print(f"[{name}] FAILED: {e}")

    # --- plots: 4 figures, one per metric, 6 lines per figure -----------------
    os.makedirs("figs", exist_ok=True)
    for m in METRICS:
        plt.figure(figsize=(7, 5))
        for name, v in per_method_val.items():
            if m in v.columns:
                plt.plot(v.index, v[m], label=name)
        plt.xlabel("step")
        plt.ylabel(m)
        plt.title(m)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out = f"figs/{m}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"saved {out}")

    # --- averaged accuracy table ---------------------------------------------
    if per_method_acc:
        table = pd.DataFrame(per_method_acc).T
        table.index.name = "method"
        table.to_csv("accuracy_table.csv")
        print(f"saved accuracy_table.csv, shape={table.shape}")
        print(table)


if __name__ == "__main__":
    main()
