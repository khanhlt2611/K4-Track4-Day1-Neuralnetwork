"""Pipeline MLP: seed, loss, metrics, huấn luyện, dự đoán và CSV.

run_experiment chỉ đọc train/val. Metric summary lấy ở epoch có val_loss
thấp nhất; final_*_loss là loss ở epoch cuối hoàn thành. best_state giữ
trong RAM, JSON chỉ lưu cfg/history/summary, không lưu checkpoint.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients


DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce", optimizer="sgd_momentum", lr=None,
    weight_decay=0.0, momentum=0.9, batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he", clip_norm=None,
    precision="fp32", seed=1,
)


def set_seed(seed: int) -> None:
    """Cố định RNG; tắt cuDNN benchmark để so sánh seed ổn định hơn."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """Macro-F1 trên đủ 7 lớp, kể cả lớp không xuất hiện (F1=0)."""
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (7, 7) or not np.isfinite(cm).all() or (cm < 0).any():
        raise ValueError("cm phải là ma trận 7x7 hữu hạn và không âm")
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(7), where=denom > 0)
    return float(f1.mean())


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """CE trên logits; MSE logits-vs-one-hot trung bình trên 7 lớp/mẫu.

    reduction='sum' cộng loss từng mẫu (MSE đã chia 7), để evaluate chia
    đúng cho N và không phụ thuộc kích thước batch cuối.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction phải là 'mean', 'sum' hoặc 'none'")
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name != "mse":
        raise ValueError("loss phải là 'ce' hoặc 'mse'")
    target = F.one_hot(y, num_classes=7).to(dtype=logits.dtype)
    per_sample = F.mse_loss(logits, target, reduction="none").mean(dim=1)
    if reduction == "none":
        return per_sample
    return per_sample.mean() if reduction == "mean" else per_sample.sum()


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Argmax logits, giữ thứ tự; trả về int64 trên model device."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương")
    device = next(model.parameters()).device
    model.eval()
    if len(X) == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.cat([model(X[start:start + batch_size].to(device)).argmax(dim=1)
                      for start in range(0, len(X), batch_size)])


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Loss theo mẫu và metrics FP32 ở eval mode, dropout tắt."""
    if len(X) != len(y) or len(y) == 0:
        raise ValueError("X/y phải cùng số mẫu và không rỗng")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương")
    device = next(model.parameters()).device
    model.eval()
    cm = torch.zeros(7, 7, dtype=torch.int64, device=device)
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, len(y), batch_size):
        xb = X[start:start + batch_size].to(device)
        yb = y[start:start + batch_size].to(device)
        logits = model(xb)
        total_loss += compute_loss(logits, yb, loss_name, reduction="sum").double()
        cm += torch.bincount(yb * 7 + logits.argmax(dim=1), minlength=49).reshape(7, 7)
    confusion = cm.cpu().numpy()
    return dict(loss=total_loss.item() / len(y), acc=float(np.trace(confusion) / len(y)),
                macro_f1=macro_f1_from_confusion(confusion))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _json_safe(value):
    """Run diverged vẫn có JSON hợp lệ: số không hữu hạn biểu diễn bằng null."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_training_result(result: dict, path: str | Path) -> None:
    """Lưu cfg/history/summary; bỏ best_state và checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: result[key] for key in ("cfg", "history", "summary")}
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Train/val, log từng epoch và giữ bản sao best_state trên CPU.

    cfg tuỳ chọn: train_eval_size=50_000 (None=toàn bộ train), eval_batch=8192,
    scheduler=None/'cosine', scheduler_kwargs={}, verbose=True. Tập con đo
    train loss có seed 42 cố định giữa thí nghiệm và các seed huấn luyện.
    FP16 luôn unscale trước đo/clip; overflow gradient bỏ bước qua GradScaler.
    Loss NaN/inf hoặc gradient FP32/BF16 NaN/inf đánh dấu diverged và dừng.
    Không đọc bất kỳ khoá eval nào trong data.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    for key in ("batch", "epochs", "eval_batch"):
        value = cfg.get(key, 8192)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} phải là số nguyên dương")
    hidden = tuple(cfg["hidden"])
    if hidden not in EXPECTED_PARAMS:
        raise ValueError("hidden phải là M-base, M-wide hoặc M-deep theo đề")
    if cfg["loss"] not in ("ce", "mse"):
        raise ValueError("loss phải là 'ce' hoặc 'mse'")
    precision = cfg["precision"]
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision phải là 'fp32', 'fp16' hoặc 'bf16'")
    max_norm = cfg["clip_norm"]
    if max_norm is not None and (not math.isfinite(max_norm) or max_norm <= 0):
        raise ValueError("clip_norm phải hữu hạn và dương, hoặc None")
    X_tr, y_tr, X_val, y_val = (data[k] for k in ("X_tr", "y_tr", "X_val", "y_val"))
    for X, y in ((X_tr, y_tr), (X_val, y_val)):
        if X.ndim != 2 or X.shape[1] != 54 or len(y) != len(X) or len(y) == 0:
            raise ValueError("Dữ liệu cần X=(N,54), y=(N,), N>0")
        if X.dtype != torch.float32 or y.dtype != torch.int64 or y.ndim != 1:
            raise ValueError("X phải float32 và y phải int64 một chiều")
        if X.device != y.device or ((y < 0) | (y > 6)).any().item():
            raise ValueError("X/y phải cùng device, nhãn phải thuộc 0..6")
    device = X_tr.device
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 trong pipeline này cần CUDA + GradScaler")
    if precision == "bf16" and device.type == "cuda":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("GPU không hỗ trợ BF16; ghi lý do vào notes")
    if precision != "fp32" and device.type not in ("cpu", "cuda"):
        raise ValueError("Mixed precision chỉ hỗ trợ CPU/CUDA")
    set_seed(cfg["seed"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden]
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"],
                                betas=cfg.get("betas", (0.9, 0.999)), eps=cfg.get("eps", 1e-8))
    total_steps = math.ceil(len(y_tr) / cfg["batch"]) * cfg["epochs"]
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), total_steps,
                                **cfg.get("scheduler_kwargs", {}))
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    eval_batch = cfg.get("eval_batch", 8192)
    sample_size = cfg.get("train_eval_size", 50_000)
    if sample_size is not None and (not isinstance(sample_size, int) or sample_size <= 0):
        raise ValueError("train_eval_size phải là số nguyên dương, hoặc None")
    if sample_size is not None and sample_size < len(y_tr):
        ids = np.random.default_rng(42).choice(len(y_tr), sample_size, replace=False)
        indices = torch.as_tensor(ids, dtype=torch.int64, device=device)
        X_monitor, y_monitor = X_tr[indices], y_tr[indices]
    else:
        X_monitor, y_monitor = X_tr, y_tr
    cfg["train_eval_samples"] = len(y_monitor)
    cfg["train_eval_seed"] = 42
    history = {key: [] for key in ("epoch", "train_loss", "val_loss", "val_acc",
                "val_macro_f1", "grad_norm", "epoch_time_s", "lr", "optimizer_steps",
                "amp_skipped_steps", "grad_norm_max", "clip_fraction")}
    if device.type == "cuda":
        _sync(device)
        torch.cuda.reset_peak_memory_stats(device)
    initial = evaluate(model, X_val, y_val, cfg["loss"], eval_batch)
    step0_loss = initial["loss"]
    diverged = not math.isfinite(step0_loss)
    reason = "nonfinite_step0_loss" if diverged else None
    best_loss, best_epoch, best_state, best_metrics = math.inf, None, None, None
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    verbose = cfg.get("verbose", True)
    if verbose:
        print(f"{cfg['exp_id']} | {device} | {precision} | step0 loss={step0_loss:.6f}")
    for epoch in range(1, cfg["epochs"] + 1):
        if diverged:
            break
        _sync(device)
        start_time = time.perf_counter()
        model.train()
        norms, updated, skipped, clipped = [], 0, 0, 0
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=generator):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "fp32"):
                loss = compute_loss(model(xb), yb, cfg["loss"])
            if not torch.isfinite(loss).item():
                diverged, reason = True, "nonfinite_train_loss"
                break
            scaler.scale(loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)  # cũng cần khi không clip để log gradient thật
            norm = clip_gradients(model.parameters(), max_norm)
            if not math.isfinite(norm):
                if scaler.is_enabled():
                    scaler.step(optimizer)  # tự bỏ bước chứa gradient overflow
                    scaler.update()
                    skipped += 1
                    continue
                diverged, reason = True, "nonfinite_gradient"
                break
            norms.append(norm)
            clipped += int(max_norm is not None and norm > max_norm)
            scaler.step(optimizer)
            scaler.update()
            updated += 1
            if scheduler is not None:
                scheduler.step()
        # Không ghi một epoch dở dang như một epoch đã hoàn thành.
        if diverged:
            break
        train_metrics = evaluate(model, X_monitor, y_monitor, cfg["loss"], eval_batch)
        val_metrics = evaluate(model, X_val, y_val, cfg["loss"], eval_batch)
        _sync(device)
        elapsed = time.perf_counter() - start_time
        if not all(math.isfinite(m["loss"]) for m in (train_metrics, val_metrics)):
            diverged, reason = True, "nonfinite_epoch_loss"
            break
        values = dict(epoch=epoch, train_loss=train_metrics["loss"], val_loss=val_metrics["loss"],
                      val_acc=val_metrics["acc"], val_macro_f1=val_metrics["macro_f1"],
                      grad_norm=float(np.mean(norms)) if norms else None, epoch_time_s=elapsed,
                      lr=optimizer.param_groups[0]["lr"], optimizer_steps=updated,
                      amp_skipped_steps=skipped, grad_norm_max=max(norms) if norms else None,
                      clip_fraction=clipped / len(norms) if norms else 0.0)
        for key, value in values.items():
            history[key].append(value)
        if val_metrics["loss"] < best_loss:
            best_loss, best_epoch, best_metrics = val_metrics["loss"], epoch, val_metrics.copy()
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        if verbose:
            print(f"epoch {epoch:02d}/{cfg['epochs']} | train={values['train_loss']:.4f} "
                  f"val={values['val_loss']:.4f} acc={values['val_acc']:.4f} "
                  f"macro-F1={values['val_macro_f1']:.4f} "
                  f"grad={values['grad_norm']} time={elapsed:.2f}s skipped={skipped}")
    summary = dict(step0_loss=step0_loss,
                   best_val_loss=best_loss if best_epoch is not None else None,
                   best_epoch=best_epoch,
                   final_train_loss=history["train_loss"][-1] if history["epoch"] else None,
                   final_val_loss=history["val_loss"][-1] if history["epoch"] else None,
                   val_acc=best_metrics["acc"] if best_metrics else None,
                   val_macro_f1=best_metrics["macro_f1"] if best_metrics else None,
                   time_per_epoch_s=float(np.mean(history["epoch_time_s"])) if history["epoch"] else None,
                   peak_mem_MB=torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
                   diverged=diverged, divergence_reason=reason,
                   completed_epochs=len(history["epoch"]),
                   amp_skipped_steps=sum(history["amp_skipped_steps"]))
    if verbose and diverged:
        print(f"Stopped: {reason}; completed epochs={len(history['epoch'])}")
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """CSV row_id,pred: nhãn nguyên 0..6 và row_id nguyên duy nhất."""
    arrays = []
    for values in (row_id, preds):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
            raise ValueError("row_id/pred phải là mảng số một chiều")
        if not np.isfinite(values).all() or not np.equal(values, values.astype(np.int64)).all():
            raise ValueError("row_id/pred phải chứa số nguyên hữu hạn")
        arrays.append(values.astype(np.int64))
    ids, labels = arrays
    if len(ids) == 0 or len(ids) != len(labels) or len(np.unique(ids)) != len(ids):
        raise ValueError("CSV cần row_id duy nhất, không rỗng, cùng số lượng pred")
    if (ids < 0).any() or ((labels < 0) | (labels > 6)).any():
        raise ValueError("row_id phải không âm, pred phải thuộc 0..6")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "pred"])
        writer.writerows(zip(ids.tolist(), labels.tolist()))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Nạp best_state và xuất CSV; chạy evaluate.py riêng sau chọn bằng val."""
    if result.get("best_state") is None:
        raise ValueError("Run chưa có best_state hợp lệ để dự đoán")
    effective = result["cfg"]
    for key in ("hidden", "dropout", "init"):
        requested, actual = cfg.get(key, effective[key]), effective[key]
        if key == "hidden":
            requested, actual = tuple(requested), tuple(actual)
        if requested != actual:
            raise ValueError(f"cfg[{key}] không khớp result")
    model = MLP(hidden=effective["hidden"], dropout=effective["dropout"], init=effective["init"])
    model.load_state_dict(result["best_state"])
    model.to(data["X_eval"].device)
    write_predictions(data["eval_row_id"], predict(model, data["X_eval"]), pred_path)


def main() -> None:
    """CLI chạy một cấu hình train/val, không tự chấm eval."""
    import argparse
    from health_checks import load_training_data

    default_root = next(p for p in Path(__file__).resolve().parents
                        if (p / "scripts/split_data.py").exists())
    default_out = Path(__file__).resolve().parents[1]
    if default_out == default_root:
        default_out = default_root / "submission_2A202602941"

    parser = argparse.ArgumentParser(
        description="Train an MLP using train/validation; save JSON history without scoring eval.")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--optimizer", default="sgd_momentum")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--clip-norm", type=float)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exp-id", default="base-s1")
    parser.add_argument("--group", default="baseline")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    args = parser.parse_args()
    if Path(args.exp_id).name != args.exp_id or args.exp_id in (".", ".."):
        raise ValueError("exp_id phải là một tên file đơn")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    data = {key: value.to(args.device) for key, value in load_training_data(args.repo_root).items()}
    cfg = {**DEFAULT_CFG, "lr": args.lr, "epochs": args.epochs, "optimizer": args.optimizer,
           "precision": args.precision, "clip_norm": args.clip_norm,
           "seed": args.seed, "exp_id": args.exp_id, "group": args.group}
    result = run_experiment(cfg, data)
    save_training_result(result, args.out_dir / "results" / f"{args.exp_id}.json")
    print(json.dumps(_json_safe(result["summary"]), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
