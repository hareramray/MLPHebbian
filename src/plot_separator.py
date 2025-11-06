import argparse
import csv
import os
from typing import List, Tuple

import numpy as np


def read_csv_xy(csv_path: str, x_name: str, y_name: str, label_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        name_to_idx = {n: i for i, n in enumerate(header)}
        if x_name not in name_to_idx or y_name not in name_to_idx or label_name not in name_to_idx:
            raise ValueError(f"CSV {csv_path} must contain columns: {x_name}, {y_name}, {label_name}. Found: {header}")
        xi = name_to_idx[x_name]
        yi = name_to_idx[y_name]
        li = name_to_idx[label_name]
        Xs = []
        Ys = []
        Ls = []
        for row in r:
            try:
                Xs.append(float(row[xi]))
                Ys.append(float(row[yi]))
                Ls.append(int(float(row[li])))
            except Exception:
                # skip malformed
                continue
        X = np.asarray(Xs, dtype=np.float32)
        Y = np.asarray(Ys, dtype=np.float32)
        L = np.asarray(Ls, dtype=np.int64)
        return X, Y, L, header


def perceptron_fit(X: np.ndarray, Y: np.ndarray, epochs: int = 25, lr: float = 0.1) -> np.ndarray:
    """
    Fit a 2D perceptron on standardized inputs using labels in {-1, +1}.
    X: shape (N, 2) standardized
    Y: shape (N,) values in {-1, +1}
    Returns w of shape (3,) including bias term.
    """
    N = X.shape[0]
    Xb = np.c_[np.ones((N, 1), dtype=X.dtype), X]  # bias, x1, x2
    w = np.zeros(3, dtype=X.dtype)
    rng = np.random.default_rng(123)
    idx = np.arange(N)
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            xi = Xb[i]
            yi = Y[i]
            margin = yi * np.dot(w, xi)
            if margin <= 0:
                w = w + lr * yi * xi
    return w


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logistic_fit(
    X: np.ndarray,
    y01: np.ndarray,
    epochs: int = 200,
    lr: float = 0.1,
    l2: float = 0.0,
) -> np.ndarray:
    """
    Fit 2D logistic regression using batch gradient descent.
    X: shape (N, 2) standardized (or raw if standardization disabled)
    y01: shape (N,) in {0,1}
    Returns w of shape (3,) including bias term.
    """
    N = X.shape[0]
    Xb = np.c_[np.ones((N, 1), dtype=X.dtype), X]
    w = np.zeros(3, dtype=X.dtype)
    for _ in range(epochs):
        z = Xb @ w
        p = sigmoid(z)
        # Gradient of cross-entropy with L2 on weights (exclude bias from L2)
        grad = (Xb.T @ (p - y01)) / N
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def boundary_y_from_x_raw(x_raw: np.ndarray, w: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Given raw-space x values (for feature 0), compute raw-space y values (feature 1)
    for the decision boundary w0 + w1*x1_std + w2*x2_std = 0.
    std and mean vectors correspond to [x1, x2].
    """
    w0, w1, w2 = w
    if abs(w2) < 1e-8:
        # Vertical boundary; caller should handle separately
        return np.full_like(x_raw, np.nan)
    x1_std = (x_raw - mean[0]) / (std[0] if std[0] != 0 else 1.0)
    x2_std = -(w0 + w1 * x1_std) / w2
    y_raw = x2_std * (std[1] if std[1] != 0 else 1.0) + mean[1]
    return y_raw


def main():
    ap = argparse.ArgumentParser(description="Scatter plot with perceptron separator (health dataset)")
    ap.add_argument("--data", type=str, default=os.path.join("data", "health_risk.csv"))
    ap.add_argument("--x", type=str, default="bmi", help="X-axis column name (e.g., bmi)")
    ap.add_argument("--y", type=str, default="steps_per_day", help="Y-axis column name (e.g., steps_per_day)")
    ap.add_argument("--label", type=str, default="high_risk", help="Binary label column name (0/1)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.1)
    # Logistic regression options
    ap.add_argument("--use-logistic", action="store_true", help="Also fit logistic regression and draw its 0.5 boundary")
    ap.add_argument("--log-epochs", type=int, default=200, help="Logistic regression training epochs")
    ap.add_argument("--log-lr", type=float, default=0.1, help="Logistic regression learning rate")
    ap.add_argument("--log-l2", type=float, default=0.0, help="Logistic regression L2 regularization strength")
    ap.add_argument("--no-standardize", action="store_true", help="Disable feature standardization before perceptron training")
    ap.add_argument("--out", type=str, default=os.path.join("plots", "health_separator.png"))
    args = ap.parse_args()

    # Ensure data exists; if not, try to generate it
    if not os.path.exists(args.data):
        try:
            from health_data_generator import generate_health_dataset, save_csv  # type: ignore
            X, risk_score, y = generate_health_dataset(1000)
            os.makedirs(os.path.dirname(args.data), exist_ok=True)
            save_csv(args.data, X, risk_score, y)
            print(f"Generated dataset at {args.data}")
        except Exception as e:
            print(f"[warn] Could not auto-generate data: {e}")

    X1, X2, L, header = read_csv_xy(args.data, args.x, args.y, args.label)
    # Prepare training arrays
    X_raw = np.stack([X1, X2], axis=1)
    y01 = (L > 0).astype(np.int64)
    y_pm = (2 * y01 - 1).astype(np.int64)  # {-1, +1}

    # Standardize if enabled
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0) + 1e-8
    X_std = (X_raw - mean) / std if not args.no_standardize else X_raw.copy()

    # Train perceptron
    w = perceptron_fit(X_std, y_pm, epochs=args.epochs, lr=args.lr)
    # Train logistic regression if requested
    w_log = None
    if args.use_logistic:
        w_log = logistic_fit(X_std, y01, epochs=args.log_epochs, lr=args.log_lr, l2=args.log_l2)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.figure(figsize=(6.5, 5))

    # Scatter points in raw units
    pos = y01 == 1
    neg = ~pos
    plt.scatter(X_raw[neg, 0], X_raw[neg, 1], s=12, c="#1f77b4", alpha=0.7, label="no risk (0)")
    plt.scatter(X_raw[pos, 0], X_raw[pos, 1], s=12, c="#d62728", alpha=0.7, label="high risk (1)")

    # Decision boundary
    w0, w1, w2 = w
    # Compute a margin around observed x range
    x_min, x_max = X_raw[:, 0].min(), X_raw[:, 0].max()
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    xs = np.linspace(x_min - x_pad, x_max + x_pad, 200)

    if abs(w2) < 1e-8:
        # Vertical boundary in standardized space -> vertical in raw space
        x_std_zero = -w0 / (w1 if abs(w1) > 1e-8 else 1e-8)
        x_vert = x_std_zero * (std[0] if not args.no_standardize else 1.0) + (mean[0] if not args.no_standardize else 0.0)
        plt.axvline(x_vert, color="black", linestyle="--", linewidth=1.2, label="perceptron boundary")
    else:
        ys = boundary_y_from_x_raw(xs, w, mean if not args.no_standardize else np.zeros(2), std if not args.no_standardize else np.ones(2))
        plt.plot(xs, ys, color="black", linestyle="--", linewidth=1.2, label="perceptron boundary")

    # Logistic boundary at p=0.5 (z=0)
    if w_log is not None:
        w0l, w1l, w2l = w_log
        if abs(w2l) < 1e-8:
            x_std_zero = -w0l / (w1l if abs(w1l) > 1e-8 else 1e-8)
            x_vert = x_std_zero * (std[0] if not args.no_standardize else 1.0) + (mean[0] if not args.no_standardize else 0.0)
            plt.axvline(x_vert, color="#2ca02c", linestyle="-", linewidth=1.4, label="logistic p=0.5")
        else:
            ys_log = boundary_y_from_x_raw(xs, w_log, mean if not args.no_standardize else np.zeros(2), std if not args.no_standardize else np.ones(2))
            plt.plot(xs, ys_log, color="#2ca02c", linestyle="-", linewidth=1.4, label="logistic p=0.5")

    plt.xlabel(args.x)
    plt.ylabel(args.y)
    title_extra = " + logistic" if w_log is not None else ""
    plt.title(f"Decision boundary: {args.x} vs {args.y}{title_extra}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"Saved separator plot to {args.out}")


if __name__ == "__main__":
    main()
