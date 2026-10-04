"""Vẽ log huấn luyện và so sánh các thí nghiệm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg, history = result["cfg"], result["history"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = history.get("epoch", [])
    if epochs:
        axes[0].plot(epochs, history["train_loss"], label="Train (eval mode)")
        axes[0].plot(epochs, history["val_loss"], label="Validation")
        axes[1].plot(epochs, history["val_acc"], label="Val accuracy")
        axes[1].plot(epochs, history["val_macro_f1"], label="Val macro-F1")
        axes[2].plot(epochs, history["grad_norm"], label="Before clipping")
        best = result["summary"].get("best_epoch")
        for ax in axes:
            if best is not None:
                ax.axvline(best, color="gray", linestyle="--", alpha=0.6)
            ax.legend(fontsize=9)
    else:
        reason = result["summary"].get("skip_reason") or result["summary"].get("divergence_reason", "No completed epoch")
        for ax in axes:
            ax.text(0.5, 0.5, reason, ha="center", va="center", wrap=True, transform=ax.transAxes)
    for ax, label in zip(axes, ("Loss", "Score", "Mean gradient L2 norm")):
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(alpha=0.25)
    axes[1].set_ylim(0, 1)
    fig.suptitle(f"{cfg['exp_id']} | {cfg['optimizer']} lr={cfg['lr']} | batch={cfg['batch']} | "
                 f"hidden={cfg['hidden']} dropout={cfg['dropout']} | {cfg['precision']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    count = 0
    for result in results:
        history = result["history"]
        if history.get("epoch") and metric in history:
            ax.plot(history["epoch"], history[metric], label=result["cfg"]["exp_id"])
            count += 1
    if count:
        ax.legend(fontsize=8)
    ax.set(xlabel="Epoch", ylabel=metric, title=title or metric)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(eval_result: dict, path: str) -> None:
    """Ma trận nhầm lẫn lấy nguyên số liệu của script chấm chính thức."""
    import numpy as np
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(eval_result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted class", ylabel="True class", title="Final eval confusion matrix",
           xticks=range(7), yticks=range(7))
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
