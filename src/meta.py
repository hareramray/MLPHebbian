from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import copy
import os
import sys
import numpy as np

# Allow running as a script without package context
sys.path.append(os.path.dirname(__file__))

from mlp_core import MLP, HebbianConfig, OptimConfig


@dataclass
class MetaSearchSpace:
    lrs: List[float]
    alphas: List[float]


class MetaTuner:
    """
    A tiny, brute-force meta tuner over learning rate (lr) and Hebbian alpha.

    Strategy per epoch:
      - Snapshot model weights
      - For each candidate (lr, alpha):
          - Clone model
          - Run a few gradient steps on a small training subset (warmup_steps)
          - Evaluate on validation subset
      - Pick best hyperparams this epoch
      - Update the real model's optimizer params
    """
    def __init__(self, search: MetaSearchSpace, warmup_steps: int = 1, batch_size: int = 64):
        self.search = search
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size

    def _mini_train(self, model: MLP, X: np.ndarray, y: np.ndarray, task: str) -> None:
        # just a few steps on a small subset for quick estimation
        n = min(len(X), self.batch_size)
        idx = np.random.choice(len(X), size=n, replace=False)
        xb = X[idx]
        yb = y[idx]
        zs, ys = model.forward(xb)
        if task == "binary":
            loss, grad_out = model.bce_with_logits(zs[-1], yb)
        else:
            loss, grad_out = model.mse(ys[-1], yb)
        dWs, dbs = model.backward(zs, ys, grad_out)
        # re-use ys for hebbian pre/post
        model._step(dWs, dbs, ys, ys)

    def suggest(self, model: MLP, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, task: str) -> Tuple[float, float]:
        # Snapshot state
        base_state = model._state_dict()
        best_val = float('inf')
        best = (model.optim.lr, model.hebb.alpha)
        for lr in self.search.lrs:
            for alpha in self.search.alphas:
                # clone model
                clone = copy.deepcopy(model)
                clone.optim.lr = lr
                clone.hebb.alpha = alpha
                # do a couple mini updates
                for _ in range(self.warmup_steps):
                    self._mini_train(clone, X_train, y_train, task)
                # evaluate
                val_loss = clone.evaluate_loss(X_val, y_val, task)
                if val_loss < best_val:
                    best_val = val_loss
                    best = (lr, alpha)
        # restore real model weights
        model._load_state(base_state)
        return best
