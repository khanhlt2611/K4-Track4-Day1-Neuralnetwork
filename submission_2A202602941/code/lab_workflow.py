"""Reusable experiment logging, comparison and official final evaluation."""
from collections import defaultdict
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from model import MLP, activation_stats
from train import set_seed

from train import run_experiment, final_eval
from plots import plot_run, plot_compare, plot_confusion
from results_table import save_result, to_row, write_xlsx


def run_logged(cfg, data, out_dir):
    print("Prediction before run:", cfg.get("prediction", "Loss should decrease; compare validation scores."))
    cfg = dict(cfg)
    if cfg.get("group") == "init":
        set_seed(cfg["seed"])
        probe = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"]).to(data["X_tr"].device)
        cfg["activation_std"] = activation_stats(probe, data["X_tr"][:2048])
        print("Activation std after each Linear:", cfg["activation_std"])
        del probe
    result = run_experiment(cfg, data)
    save_result(result, str(Path(out_dir) / "results"))
    plot_run(result, str(Path(out_dir) / "figures" / f"{cfg['exp_id']}.png"))
    return result


def valid_results(results):
    return [r for r in results if not r["summary"]["diverged"]
            and r["best_state"] is not None and r["summary"]["val_macro_f1"] is not None]


def select_by_val(results):
    candidates = valid_results(results)
    if not candidates:
        raise RuntimeError("No successful run. Inspect divergence_reason and learning rates.")
    # Do not compare raw CE/MSE loss to break ties across different loss scales.
    return max(candidates, key=lambda r: r["summary"]["val_macro_f1"])


def seed_statistics(results):
    candidates = valid_results(results)
    stats = {}
    for key in ("val_acc", "val_macro_f1", "best_val_loss"):
        values = np.array([r["summary"][key] for r in candidates])
        stats[key] = {"mean": float(values.mean()) if len(values) else None,
                      "std": float(values.std(ddof=1)) if len(values) > 1 else None}
    stats["n"] = len(candidates)
    std = stats["val_macro_f1"]["std"]
    stats["noise_2sigma"] = 2 * std if std is not None else None
    return stats


def summarize_runs(results, baseline, noise, out_dir):
    reference = baseline["summary"]["val_macro_f1"]
    rows = []
    grouped = defaultdict(list)
    for result in results:
        cfg, summary = result["cfg"], result["summary"]
        score = summary["val_macro_f1"]
        delta = score - reference if score is not None else None
        observation = (f"val macro-F1={score:.6f}; delta vs base seed1={delta:+.6f}; "
                       + (f"|delta| > 2sigma: {abs(delta) > noise}" if noise is not None
                          else "seed noise unavailable")) if score is not None else "No completed epoch"
        cfg["observation"] = observation
        save_result(result, str(Path(out_dir) / "results"))
        print(cfg["exp_id"], observation)
        rows.append({"exp_id": cfg["exp_id"], "group": cfg["group"], "lr": cfg["lr"],
                     "val_macro_f1": score, "val_acc": summary["val_acc"],
                     "best_epoch": summary["best_epoch"], "delta_vs_base": delta,
                     "diverged": summary["diverged"]})
        grouped[cfg["group"]].append(result)
    for group, runs in grouped.items():
        plot_compare(runs, "val_macro_f1", str(Path(out_dir) / "figures" / f"compare_{group}.png"))
    return pd.DataFrame(rows)


def score_final(result, data, repo_root, out_dir, baseline=False):
    output = Path(out_dir)
    stem = "baseline" if baseline else "eval"
    pred = output / ("predictions_baseline.csv" if baseline else "predictions_eval.csv")
    scores_path = output / ("baseline_eval_result.json" if baseline else "eval_result.json")
    final_eval(result["cfg"], result, data, str(pred))
    # Explicit data paths work even when outputs reside outside the repository.
    subprocess.run([sys.executable, str(Path(repo_root) / "scripts/evaluate.py"),
                    "--data", str(Path(repo_root) / "data/covtype.csv.gz"),
                    "--meta", str(Path(repo_root) / "data/split_metadata.csv"),
                    "--pred", str(pred), "--out", str(scores_path)], check=True)
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    if not baseline:
        plot_confusion(scores, str(output / "figures/confusion_eval.png"))
    return scores


def export_table(results, scores_by_id, repo_root, out_dir):
    rows = [to_row(r, scores_by_id.get(r["cfg"]["exp_id"])) for r in results]
    target = Path(out_dir) / "experiments.xlsx"
    write_xlsx(rows, str(Path(repo_root) / "templates/experiment_table_template.xlsx"), str(target))
    return target
