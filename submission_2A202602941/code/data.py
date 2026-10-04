"""Nạp dữ liệu, chia validation, chuẩn hoá bằng train và chia batch.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    with np.load(f"{processed_dir}/train.npz") as train:
        X_train_full, y_train_full = train["X"], train["y"]
    with np.load(f"{processed_dir}/eval.npz") as evaluation:
        X_eval, y_eval = evaluation["X"], evaluation["y"]
        eval_row_id = evaluation["row_id"]

    for X, y in ((X_train_full, y_train_full), (X_eval, y_eval)):
        assert X.ndim == 2 and X.shape[1] == 54
        assert X.dtype == np.float32
        assert y.shape == (X.shape[0],)
        assert y.dtype == np.int64
    assert eval_row_id.shape == (X_eval.shape[0],)

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    numeric = X_tr[:, :N_NUMERIC]
    return numeric.mean(axis=0, dtype=np.float64), numeric.std(axis=0, dtype=np.float64)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    standardized = X.copy()
    numeric = standardized[:, :N_NUMERIC]
    numeric -= mean
    numeric /= np.where(std == 0, 1, std)
    return standardized


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )
    mean, std = fit_standardizer(X_tr)

    majority_class = np.bincount(y_tr).argmax()
    majority_accuracy = np.mean(y_val == majority_class)
    print(f"Split sizes: train={len(y_tr)}, val={len(y_val)}, eval={len(y_eval)}")
    print(f"Majority-class validation accuracy: {majority_accuracy:.4f}")

    return {
        "X_tr": torch.as_tensor(apply_standardizer(X_tr, mean, std), dtype=torch.float32, device=device),
        "y_tr": torch.as_tensor(y_tr, dtype=torch.int64, device=device),
        "X_val": torch.as_tensor(apply_standardizer(X_val, mean, std), dtype=torch.float32, device=device),
        "y_val": torch.as_tensor(y_val, dtype=torch.int64, device=device),
        "X_eval": torch.as_tensor(apply_standardizer(X_eval, mean, std), dtype=torch.float32, device=device),
        "y_eval": torch.as_tensor(y_eval, dtype=torch.int64, device=device),
        "eval_row_id": eval_row_id,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương")
    if len(X) != len(y) or X.device != y.device:
        raise ValueError("X và y phải cùng số mẫu và cùng device")
    if shuffle:
        perm = torch.randperm(len(X), generator=generator, device=X.device)
    else:
        perm = torch.arange(len(X), device=X.device)
    for i in range(0, len(X), batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
