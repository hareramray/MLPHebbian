import argparse
import csv
import os
from typing import Tuple
import numpy as np

RNG = np.random.default_rng()


def generate_health_dataset(n: int, noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a human-understandable health-risk dataset.

    Features (X):
      0: bmi (kg/m^2)
      1: steps_per_day
      2: daily_calories
    Targets:
      risk_score: 0-100 (derived from interpretable formula)
      high_risk: 0/1 (thresholded risk_score)
    """
    # Features with realistic ranges
    bmi = np.clip(RNG.normal(26.0, 6.0, size=n), 15.0, 45.0)
    steps = np.clip(RNG.normal(8000.0, 3000.0, size=n), 1000.0, 18000.0)
    calories = np.clip(RNG.normal(2100.0, 400.0, size=n), 1200.0, 4000.0)

    # Interpretable risk signal: higher BMI and calories increase risk; more steps reduce risk
    signal = (
        0.12 * (bmi - 22.0)              # stronger BMI influence
        + 0.00018 * (calories - 2000.0)   # stronger calorie surplus influence
        - 0.00008 * (steps - 8000.0)      # stronger protective effect of steps
    )
    # Slightly stronger quadratic for high BMI to increase separability at upper tail
    signal += 0.003 * np.maximum(bmi - 30.0, 0.0) ** 2

    # Add noise
    signal += RNG.normal(0.0, noise_std, size=n)

    # Map signal -> probability via sigmoid, then score 0-100
    prob = 1.0 / (1.0 + np.exp(-signal))
    score = 100.0 * prob

    # Slightly lower threshold to improve positive class coverage
    high_risk = (prob >= 0.55).astype(int)

    X = np.stack([bmi, steps, calories], axis=1).astype(np.float32)
    risk_score = score.astype(np.float32)
    y = high_risk.astype(np.int64)
    return X, risk_score, y


def save_csv(path: str, X: np.ndarray, risk_score: np.ndarray, high_risk: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bmi", "steps_per_day", "daily_calories", "risk_score", "high_risk"])  # header
        for i in range(X.shape[0]):
            w.writerow([
                float(X[i, 0]),
                float(X[i, 1]),
                float(X[i, 2]),
                float(risk_score[i]),
                int(high_risk[i]),
            ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--out", type=str, default=os.path.join("data", "health_risk.csv"))
    ap.add_argument("--noise-std", type=float, default=0.1)
    args = ap.parse_args()

    X, risk_score, y = generate_health_dataset(args.rows, noise_std=args.noise_std)
    save_csv(args.out, X, risk_score, y)
    print(f"Wrote {args.rows} rows to {args.out}")


if __name__ == "__main__":
    main()
