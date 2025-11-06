from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import Tuple
from datetime import datetime, timezone
import numpy as np

# Allow running as a script without package context
sys.path.append(os.path.dirname(__file__))

from mlp_core import MLP, HebbianConfig, OptimConfig
from meta import MetaTuner, MetaSearchSpace
from data_generator import generate_dataset as generate_study_dataset, save_csv as save_study_csv
from health_data_generator import generate_health_dataset, save_csv as save_health_csv


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Auto-detect schema and load features and binary label as (X, y)."""
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        X_list = []
        y_list = []
        if set(["hours_studied", "sleep_hours", "attendance_rate", "has_breakfast", "passed"]).issubset(fieldnames):
            for row in reader:
                X_list.append([
                    float(row["hours_studied"]),
                    float(row["sleep_hours"]),
                    float(row["attendance_rate"]),
                    float(row["has_breakfast"]),
                ])
                y_list.append([float(row["passed"])])
        elif set(["bmi", "steps_per_day", "daily_calories", "high_risk"]).issubset(fieldnames):
            for row in reader:
                X_list.append([
                    float(row["bmi"]),
                    float(row["steps_per_day"]),
                    float(row["daily_calories"]),
                ])
                y_list.append([float(row["high_risk"])])
        else:
            raise ValueError(f"Unrecognized CSV schema in {path}; got columns: {fieldnames}")
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def load_csv_health_multi(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load health dataset with both labels: high_risk (binary) and risk_score (regression in [0,1])."""
    X_list = []
    y_cls = []
    y_score = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X_list.append([
                float(row["bmi"]),
                float(row["steps_per_day"]),
                float(row["daily_calories"]),
            ])
            y_cls.append([float(row["high_risk"])])
            y_score.append([float(row.get("risk_score", 0.0)) / 100.0])
    return np.array(X_list, dtype=np.float32), np.array(y_cls, dtype=np.float32), np.array(y_score, dtype=np.float32)


def standardize(X: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None):
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


def split_train_val_test(X, y, train=0.7, val=0.15, seed=42):
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(train * n)
    n_val = int(val * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx])


def ensure_data(path: str, dataset: str, rows: int = 1200):
    if os.path.exists(path):
        return
    if dataset == "study":
        X, score, passed = generate_study_dataset(rows)
        save_study_csv(path, X, score, passed)
    elif dataset == "health":
        X, risk_score, high_risk = generate_health_dataset(rows)
        save_health_csv(path, X, risk_score, high_risk)
    else:
        raise ValueError("dataset must be 'study' or 'health'")
    print(f"Generated {dataset} dataset at {path}")


def accuracy_from_logits(logits: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int32)
    yb = y_true.astype(np.int32)
    correct = (preds == yb).sum()
    return float(correct) / len(y_true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, choices=["study", "health"], default="study")
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--rows", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--use-adam", action="store_true")
    ap.add_argument("--hebbian-alpha", type=float, default=0.0)
    ap.add_argument("--hebbian-decay", type=float, default=0.0)
    ap.add_argument("--use-hebbian", action="store_true")
    ap.add_argument("--use-meta", action="store_true")
    ap.add_argument("--meta-steps", type=int, default=1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--multi-task", action="store_true", help="Enable joint prediction of high_risk and risk_score (health dataset only)")
    ap.add_argument("--mt-weight", type=float, default=0.5, help="Loss weight for risk_score MSE in multi-task training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-csv", type=str, default="", help="Path to write per-epoch metrics CSV (epoch,train_loss,val_loss,lr,alpha)")
    ap.add_argument("--plot-out", type=str, default="", help="Path to save a PNG of loss curves (train/val)")
    ap.add_argument("--plot-title", type=str, default="", help="Optional plot title for the loss curves")
    ap.add_argument("--plot-weights-out", type=str, default="", help="Path to save a PNG of per-layer weight norms over epochs")
    args = ap.parse_args()

    if not args.data:
        args.data = os.path.join("data", "study_success.csv" if args.dataset == "study" else "health_risk.csv")
    ensure_data(args.data, dataset=args.dataset, rows=args.rows)

    # Load and split data
    if args.dataset == "health" and args.multi_task:
        X, y_cls, y_score = load_csv_health_multi(args.data)
        Xtrain, ytrain, Xval, yval, Xtest, ytest = split_train_val_test(X, y_cls, train=0.7, val=0.15, seed=args.seed)
        yscore_tr, _d1, yscore_val, _d2, yscore_te, _d3 = split_train_val_test(y_score, y_score, train=0.7, val=0.15, seed=args.seed)
        Xtrain, mean, std = standardize(Xtrain)
        Xval, _, _ = standardize(Xval, mean, std)
        Xtest, _, _ = standardize(Xtest, mean, std)
    else:
        X, y = load_csv(args.data)
        Xtrain, ytrain, Xval, yval, Xtest, ytest = split_train_val_test(X, y, train=0.7, val=0.15, seed=args.seed)
        Xtrain, mean, std = standardize(Xtrain)
        Xval, _, _ = standardize(Xval, mean, std)
        Xtest, _, _ = standardize(Xtest, mean, std)

    hebb = HebbianConfig(enabled=args.use_hebbian, alpha=args.hebbian_alpha, decay=args.hebbian_decay)
    optim = OptimConfig(lr=args.lr, momentum=args.momentum, use_adam=args.use_adam, weight_decay=args.weight_decay)

    # Architecture selection
    if args.dataset == "health" and args.multi_task:
        layer_sizes = [X.shape[1], 32, 16, 2]
    elif args.dataset == "health":
        layer_sizes = [X.shape[1], 32, 16, 1]
    else:
        layer_sizes = [X.shape[1], 16, 8, 1]
    activations = ["relu", "relu", "linear"]
    model = MLP(layer_sizes=layer_sizes, activations=activations, seed=args.seed, hebbian=hebb, optim=optim)

    if args.use_meta and not args.multi_task:
        tuner = MetaTuner(MetaSearchSpace(
            lrs=sorted(set([max(1e-4, args.lr/2), args.lr])),
            alphas=sorted(set([0.0, max(0.0, args.hebbian_alpha/2), args.hebbian_alpha]))
        ), warmup_steps=args.meta_steps)
    else:
        tuner = None

    best_val = float('inf')
    best_state = None
    best_epoch = None
    train_curve = []
    val_curve = []
    # Track per-layer weight norms across epochs
    weight_curves = [[] for _ in range(len(model.layers))]

    for ep in range(1, args.epochs + 1):
        if args.dataset == "health" and args.multi_task:
            # Custom multi-task epoch
            n = Xtrain.shape[0]
            idx = np.arange(n)
            np.random.shuffle(idx)
            total = 0.0
            for s in range(0, n, args.batch_size):
                bi = idx[s:s+args.batch_size]
                xb = Xtrain[bi]
                yb_cls = ytrain[bi]
                yb_score = yscore_tr[bi]

                zs, ys_acts = model.forward(xb)
                logits = zs[-1]  # (B,2)
                out = ys_acts[-1]
                cls_loss, cls_grad = model.bce_with_logits(logits[:, [0]], yb_cls)
                reg_loss, reg_grad = model.mse(out[:, [1]], yb_score)
                total_loss = cls_loss + args.mt_weight * reg_loss

                grad_out = np.zeros_like(logits)
                grad_out[:, [0]] = cls_grad
                grad_out[:, [1]] = args.mt_weight * reg_grad
                dWs, dbs = model.backward(zs, ys_acts, grad_out)
                model._step(dWs, dbs, ys_acts, ys_acts)
                total += total_loss * len(bi)

            tr_loss = total / n
            # Validation
            zs_val, ys_val = model.forward(Xval)
            logits_val = zs_val[-1]
            out_val = ys_val[-1]
            v_cls_loss, _ = model.bce_with_logits(logits_val[:, [0]], yval)
            v_reg_loss, _ = model.mse(out_val[:, [1]], yscore_val)
            val_loss = v_cls_loss + args.mt_weight * v_reg_loss
            print(f"Epoch {ep:3d} | loss {tr_loss:.4f} | val {val_loss:.4f}")
            # Record curves
            train_curve.append(float(tr_loss))
            val_curve.append(float(val_loss))
            # Record per-layer weight norms
            wnorms = [float(np.linalg.norm(layer.W)) for layer in model.layers]
            for i, wn in enumerate(wnorms):
                weight_curves[i].append(wn)
            # Optional CSV logging
            if args.log_csv:
                os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
                write_header = not os.path.exists(args.log_csv)
                with open(args.log_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        base = ["timestamp", "epoch", "train_loss", "val_loss", "lr", "hebbian_alpha"]
                        base += [f"wnorm_l{i}" for i in range(len(model.layers))]
                        w.writerow(base)
                    row = [datetime.now(timezone.utc).isoformat(), ep, tr_loss, val_loss, model.optim.lr, model.hebb.alpha]
                    row += wnorms
                    w.writerow(row)
            if val_loss < best_val:
                best_val = val_loss
                best_state = model._state_dict()
                best_epoch = ep
        else:
            if tuner is not None:
                best_lr, best_alpha = tuner.suggest(model, Xtrain, ytrain, Xval, yval, task="binary")
                model.optim.lr = float(best_lr)
                model.hebb.alpha = float(best_alpha)
            model.fit(Xtrain, ytrain, epochs=1, batch_size=args.batch_size, task="binary", X_val=Xval, y_val=yval, verbose=True)
            # Evaluate losses for curves
            train_loss = model.evaluate_loss(Xtrain, ytrain, task="binary")
            val_loss = model.evaluate_loss(Xval, yval, task="binary")
            train_curve.append(float(train_loss))
            val_curve.append(float(val_loss))
            # Record per-layer weight norms
            wnorms = [float(np.linalg.norm(layer.W)) for layer in model.layers]
            for i, wn in enumerate(wnorms):
                weight_curves[i].append(wn)
            if args.log_csv:
                os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
                write_header = not os.path.exists(args.log_csv)
                with open(args.log_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        base = ["timestamp", "epoch", "train_loss", "val_loss", "lr", "hebbian_alpha"]
                        base += [f"wnorm_l{i}" for i in range(len(model.layers))]
                        w.writerow(base)
                    row = [datetime.now(timezone.utc).isoformat(), ep, train_loss, val_loss, model.optim.lr, model.hebb.alpha]
                    row += wnorms
                    w.writerow(row)
            if val_loss < best_val:
                best_val = val_loss
                best_state = model._state_dict()
                best_epoch = ep

    if best_state is not None:
        model._load_state(best_state)

    if args.dataset == "health" and args.multi_task:
        zs_te, ys_te = model.forward(Xtest)
        logits_te = zs_te[-1]
        out_te = ys_te[-1]
        test_acc = accuracy_from_logits(logits_te[:, [0]], ytest)
        reg_mse, _ = model.mse(out_te[:, [1]], yscore_te)
        print(f"Test acc: {test_acc*100:.2f}% | Test risk_score MSE: {reg_mse:.4f}")
    else:
        test_logits = model.predict_logits(Xtest)
        test_loss = model.evaluate_loss(Xtest, ytest, task="binary")
        test_acc = accuracy_from_logits(test_logits, ytest)
        print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc*100:.2f}%")

    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "best_weights.npz")
    model.save(out_path)
    print(f"Saved best weights to {out_path}")

    # Plot curves to PNG if requested
    if args.plot_out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(os.path.dirname(args.plot_out) or ".", exist_ok=True)
            plt.figure(figsize=(7,4))
            if train_curve:
                plt.plot(range(1, len(train_curve)+1), train_curve, label="train")
            if val_curve:
                plt.plot(range(1, len(val_curve)+1), val_curve, label="val")
            if best_epoch is not None:
                plt.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best val @ {best_epoch}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(args.plot_title or "Training Curves")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.plot_out, dpi=120)
            print(f"Saved loss curves to {args.plot_out}")
        except Exception as e:
            print(f"[warn] Could not save plot to {args.plot_out}: {e}")

    # Plot per-layer weight norms if requested
    if args.plot_weights_out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(os.path.dirname(args.plot_weights_out) or ".", exist_ok=True)
            plt.figure(figsize=(7,4))
            epochs = range(1, len(weight_curves[0]) + 1) if weight_curves and weight_curves[0] else []
            for i, curve in enumerate(weight_curves):
                if curve:
                    plt.plot(epochs, curve, label=f"W[{i}] norm")
            if best_epoch is not None and epochs:
                plt.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best val @ {best_epoch}")
            plt.xlabel("Epoch")
            plt.ylabel("||W||_F")
            plt.title("Per-layer weight norms")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.plot_weights_out, dpi=120)
            print(f"Saved weight norm curves to {args.plot_weights_out}")
        except Exception as e:
            print(f"[warn] Could not save weight plot to {args.plot_weights_out}: {e}")


if __name__ == "__main__":
    main()
