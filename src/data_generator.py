import argparse
import csv
import os
from typing import Tuple
import numpy as np

RNG = np.random.default_rng()


def generate_dataset(n: int, noise_std: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a human-understandable dataset linking simple study habits to exam outcomes.

    Features (X):
      0: hours_studied (0-10)
      1: sleep_hours (4-10)
      2: attendance_rate (0.5-1.0)
      3: has_breakfast (0/1)
    Targets:
      y_score: final_score (0-100)
      y_pass: passed (0/1), threshold at 60
    """
    hours_studied = RNG.uniform(0, 10, size=n)
    sleep_hours = RNG.uniform(4, 10, size=n)
    attendance_rate = RNG.uniform(0.5, 1.0, size=n)
    has_breakfast = RNG.integers(0, 2, size=n)

    # Transparent formula for score with bounded ranges
    # Intuition: more study, good sleep, high attendance, breakfast -> higher score
    base = 5 + 6.5 * hours_studied + 4.0 * (sleep_hours - 6) + 30.0 * (attendance_rate - 0.5) + 5.0 * has_breakfast
    noise = RNG.normal(0.0, noise_std, size=n)
    score = np.clip(base + noise, 0.0, 100.0)

    passed = (score >= 60.0).astype(int)

    X = np.stack([hours_studied, sleep_hours, attendance_rate, has_breakfast], axis=1)
    y_score = score.astype(np.float32)
    y_pass = passed.astype(np.int64)
    return X.astype(np.float32), y_score, y_pass


def save_csv(path: str, X: np.ndarray, score: np.ndarray, passed: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hours_studied", "sleep_hours", "attendance_rate", "has_breakfast", "final_score", "passed"])  # header
        for i in range(X.shape[0]):
            row = [
                float(X[i, 0]),
                float(X[i, 1]),
                float(X[i, 2]),
                int(X[i, 3]),
                float(score[i]),
                int(passed[i]),
            ]
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000, help="number of rows to generate")
    ap.add_argument("--out", type=str, default=os.path.join("data", "study_success.csv"))
    ap.add_argument("--noise-std", type=float, default=5.0)
    args = ap.parse_args()

    X, y_score, y_pass = generate_dataset(args.rows, noise_std=args.noise_std)
    save_csv(args.out, X, y_score, y_pass)
    print(f"Wrote {args.rows} rows to {args.out}")


if __name__ == "__main__":
    main()
