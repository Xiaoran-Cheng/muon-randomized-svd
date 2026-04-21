"""Parse condor .out files to extract per-trial metrics (tta_val_acc, tta_test_acc,
time_seconds) written by airbench94_muon.py, then compute per-grid-point mean/std.

Each .out file corresponds to ONE sweep grid point (or one final run). Inside it,
50 (or num_trials) eval rows are printed, one per trial.

Usage (on server, inside the folder containing .out files):
    python extract_from_stdout.py                       # parse ./*.out
    python extract_from_stdout.py --dir ablation_study
    python extract_from_stdout.py --dir /path/A /path/B --out /tmp/combined

Outputs two CSVs in --out (default: current dir):
    per_trial.csv    : variant, grid_point, trial_idx, tta_val_acc, tta_test_acc, time_seconds, ...
    summary.csv      : variant, grid_point, n_trials, mean/std of the three metrics
"""
import argparse
import csv
import glob
import os
import re
import statistics
from collections import defaultdict


# Must stay in sync with FOLDER_ARGS / ARG_ABBREVIATIONS in airbench94_muon.py.
FOLDER_FIELD_ORDER = [
    ("optimizer_mode", "om"),
    ("batch_size", "bs"),
    ("muon_lr", "mlr"),
    ("muon_momentum", "mmm"),
    ("muon_nesterov", "mn"),
    ("inexact_solver", "is"),
    ("orth_steps", "os"),
    ("randomized", "rz"),
    ("rank", "rk"),
    ("oversampling", "ov"),
    ("power_iters", "pi"),
]

# When variant name starts with one of these prefixes, the listed field is the
# swept value (used as grid_point label). Otherwise grid_point = full folder name.
ABLATION_VARIANTS = {
    "e1-solver": "inexact_solver",
    "e2-rank": "rank",
    "e3-orth-steps": "orth_steps",
    "e4-batch-size": "batch_size",
}

SAVING_RE = re.compile(r"Saving outputs to:\s*(\S+)")


def unformat(raw):
    """Reverse of airbench94_muon._format_arg_value."""
    if raw == "t":
        return True
    if raw == "f":
        return False
    numeric = raw.replace("p", ".")
    if numeric.startswith("m"):
        numeric = "-" + numeric[1:]
    try:
        return int(numeric) if "." not in numeric else float(numeric)
    except ValueError:
        return raw  # string identifier (e.g. "cubic_ns_theoretical")


def parse_folder_name(folder):
    """Extract all hyperparam fields from a folder name like
    'ommuon_bs1000_mlr0p195723_mmm0p483562_mnt_iscubic_ns_theoretical_os9_rzt_rk256_ov10_pi2'."""
    fields = {}
    for idx, (name, abbr) in enumerate(FOLDER_FIELD_ORDER):
        if idx == len(FOLDER_FIELD_ORDER) - 1:
            pattern = rf"{abbr}([^/]+?)$"
        else:
            next_abbr = FOLDER_FIELD_ORDER[idx + 1][1]
            pattern = rf"{abbr}(.+?)_{next_abbr}"
        m = re.search(pattern, folder)
        if m:
            fields[name] = unformat(m.group(1))
    return fields


def parse_saving_line(out_path, default_variant):
    """Parse the path from 'Saving outputs to: <path>'.
    Handles both layouts:
      new: /.../logs/<variant>/<folder>
      old: /.../logs/<folder>          (folder starts with 'om' and contains '_bs')
    Returns (variant, folder_name). Falls back to default_variant when variant missing."""
    parts = out_path.replace("\\", "/").rstrip("/").split("/")
    try:
        logs_idx = parts.index("logs")
    except ValueError:
        return default_variant, None
    after = parts[logs_idx + 1:]
    if not after:
        return default_variant, None
    if len(after) == 1:
        return default_variant, after[0]
    first = after[0]
    # Heuristic: real variant names never start with 'om'; folder names always do.
    if first.startswith("om") and "_bs" in first:
        return default_variant, first
    return first, after[1]


def parse_out_file(path, default_variant):
    """Return (variant, folder_name, fields_dict, list_of_eval_rows).
    Returns None if the 'Saving outputs to' header is missing."""
    variant = folder_name = None
    fields = {}
    trials = []
    current_run = None   # last non-empty value of the `run` column (sticky across rows)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if variant is None:
                m = SAVING_RE.search(line)
                if m:
                    variant, folder_name = parse_saving_line(m.group(1), default_variant)
                    if folder_name:
                        fields = parse_folder_name(folder_name)
                continue

            if not line.lstrip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 7:
                continue
            run_label, epoch_label = cells[0], cells[1]

            # The `run` column prints the trial index on the first row of each
            # trial only; subsequent rows (including the eval row) leave it blank.
            if run_label:
                current_run = run_label

            if epoch_label != "eval":
                continue
            if current_run is None or current_run == "warmup":
                continue
            try:
                trial_idx = int(current_run)
            except ValueError:
                continue
            try:
                trials.append({
                    "trial_idx": trial_idx,
                    "train_acc": float(cells[2]),
                    "val_acc": float(cells[3]),
                    "tta_val_acc": float(cells[4]),
                    "tta_test_acc": float(cells[5]),
                    "time_seconds": float(cells[6]),
                })
            except ValueError:
                pass

    if variant is None:
        return None
    return variant, folder_name, fields, trials


def grid_point_label(variant, fields, folder_name):
    """Choose how to label this grid point in the summary table."""
    field_name = ABLATION_VARIANTS.get(variant)
    if field_name and field_name in fields:
        return str(fields[field_name])
    return folder_name


def mean_std(values):
    if not values:
        return (None, None)
    if len(values) == 1:
        return (values[0], 0.0)
    return (statistics.fmean(values), statistics.stdev(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", default=["."],
                    help="Directories containing .out files (default: current dir). "
                         "Scanned non-recursively per dir.")
    ap.add_argument("--pattern", default="*.out",
                    help="Glob pattern within each --dir (default: *.out)")
    ap.add_argument("--out", default=".", help="Output directory for CSVs (default: .)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    grouped = defaultdict(list)  # (variant, grid_point) -> list of trial dicts

    out_files = []
    for d in args.dir:
        out_files.extend(sorted(glob.glob(os.path.join(d, args.pattern))))

    if not out_files:
        print(f"No files matched {args.pattern!r} in {args.dir}")
        return

    skipped = 0
    for path in out_files:
        # Fallback variant when the path in 'Saving outputs to' lacks a variant
        # subdir (old layout, e.g. main-method runs before the logs/<variant>/
        # refactor). Use the .out file's parent directory name.
        default_variant = os.path.basename(os.path.dirname(os.path.abspath(path)))
        parsed = parse_out_file(path, default_variant)
        if parsed is None:
            skipped += 1
            print(f"[skip] no 'Saving outputs to' header: {path}")
            continue
        variant, folder, fields, trials = parsed
        if not trials:
            print(f"[warn] 0 trials parsed: {path} (variant={variant})")
        gp = grid_point_label(variant, fields, folder)
        grouped[(variant, gp)].extend(trials)

    # Write timing.csv: one row per (experiment, method) with trial-averaged time
    rows = []
    for (variant, gp), trials in sorted(grouped.items()):
        time_m, time_s = mean_std([t["time_seconds"] for t in trials])
        rows.append({
            "experiment": variant,
            "method": gp,
            "n_trials": len(trials),
            "time_per_trial_mean_s": time_m,
            "time_per_trial_std_s": time_s,
        })

    out_path = os.path.join(args.out, "timing.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "method", "n_trials",
                                           "time_per_trial_mean_s", "time_per_trial_std_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {len(rows)} rows -> {out_path}")

    if skipped:
        print(f"[info] skipped {skipped} files with no 'Saving outputs to' header")


if __name__ == "__main__":
    main()
