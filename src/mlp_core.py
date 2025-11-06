from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import numpy as np

Array = np.ndarray

# Activations

def relu(x: Array) -> Array:
    return np.maximum(0, x)

def relu_backward(grad: Array, x: Array) -> Array:
    return grad * (x > 0)

def sigmoid(x: Array) -> Array:
    # numerically stable sigmoid
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    z = np.empty_like(x)
    z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    expx = np.exp(x[neg_mask])
    z[neg_mask] = expx / (1 + expx)
    return z

def sigmoid_backward(grad: Array, out: Array) -> Array:
    # out is sigmoid(x), so derivative is out*(1-out)
    return grad * out * (1 - out)

def tanh(x: Array) -> Array:
    return np.tanh(x)

def tanh_backward(grad: Array, out: Array) -> Array:
    return grad * (1 - out * out)


def linear(x: Array) -> Array:
    return x

def linear_backward(grad: Array, out: Array) -> Array:
    return grad


ACTIVATIONS: Dict[str, Tuple[Callable[[Array], Array], Callable[[Array, Array], Array]]] = {
    "relu": (relu, relu_backward),
    "sigmoid": (sigmoid, sigmoid_backward),
    "tanh": (tanh, tanh_backward),
    "linear": (linear, linear_backward),
}


@dataclass
class Layer:
    W: Array
    b: Array
    activation: str
    # Optimizer state
    vW: Optional[Array] = None
    vb: Optional[Array] = None
    mW: Optional[Array] = None
    mb: Optional[Array] = None


@dataclass
class HebbianConfig:
    enabled: bool = False
    alpha: float = 0.0  # plasticity strength
    decay: float = 0.0  # Oja-like decay term


@dataclass
class OptimConfig:
    lr: float = 1e-2
    momentum: float = 0.9
    use_adam: bool = False
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    clip: float = 5.0  # gradient clip by value
    weight_decay: float = 0.0  # L2 regularization strength


class MLP:
    def __init__(self, layer_sizes: List[int], activations: List[str], seed: int = 42,
                 hebbian: HebbianConfig = HebbianConfig(), optim: OptimConfig = OptimConfig()):
        assert len(layer_sizes) >= 2
        assert len(activations) == len(layer_sizes) - 1
        self.rng = np.random.default_rng(seed)
        self.layers: List[Layer] = []
        self.hebb = hebbian
        self.optim = optim
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Kaiming/He for ReLU, Xavier otherwise
            if activations[i] == "relu":
                std = math.sqrt(2.0 / fan_in)
            else:
                std = math.sqrt(1.0 / fan_in)
            W = self.rng.normal(0.0, std, size=(fan_out, fan_in)).astype(np.float32)
            b = np.zeros((fan_out,), dtype=np.float32)
            layer = Layer(W=W, b=b, activation=activations[i])
            # init optimizer state
            if self.optim.use_adam:
                layer.mW = np.zeros_like(W)
                layer.mb = np.zeros_like(b)
                layer.vW = np.zeros_like(W)
                layer.vb = np.zeros_like(b)
            else:
                layer.vW = np.zeros_like(W)
                layer.vb = np.zeros_like(b)
            self.layers.append(layer)
        self._adam_t = 0

    def forward(self, x: Array) -> Tuple[List[Array], List[Array]]:
        zs: List[Array] = []
        ys: List[Array] = [x]
        h = x
        for layer in self.layers:
            z = h @ layer.W.T + layer.b
            act, _ = ACTIVATIONS[layer.activation]
            y = act(z)
            zs.append(z)
            ys.append(y)
            h = y
        return zs, ys

    @staticmethod
    def bce_with_logits(y_hat_logits: Array, y_true: Array) -> Tuple[float, Array]:
        logits = y_hat_logits
        max_term = np.maximum(logits, 0)
        loss = max_term - logits * y_true + np.log1p(np.exp(-np.abs(logits)))
        grad = sigmoid(logits) - y_true
        return float(np.mean(loss)), grad / logits.shape[0]

    @staticmethod
    def mse(y_hat: Array, y_true: Array) -> Tuple[float, Array]:
        diff = y_hat - y_true
        loss = 0.5 * np.mean(diff * diff)
        grad = diff / y_hat.shape[0]
        return float(loss), grad

    def backward(self, zs: List[Array], ys: List[Array], grad_out: Array) -> Tuple[List[Array], List[Array]]:
        dWs: List[Array] = []
        dbs: List[Array] = []
        grad = grad_out
        for li in reversed(range(len(self.layers))):
            layer = self.layers[li]
            act, act_back = ACTIVATIONS[layer.activation]
            z = zs[li]
            a_prev = ys[li]
            a = ys[li + 1]
            grad_act = act_back(grad, a if layer.activation != "relu" else z)
            dW = (grad_act.T @ a_prev)
            db = np.sum(grad_act, axis=0)
            dWs.append(dW)
            dbs.append(db)
            grad = grad_act @ layer.W
        dWs.reverse()
        dbs.reverse()
        return dWs, dbs

    def _apply_hebbian(self, W: Array, pre: Array, post: Array) -> Array:
        if not self.hebb.enabled or self.hebb.alpha <= 0.0:
            return np.zeros_like(W)
        B = pre.shape[0]
        hebb_outer = (post.T @ pre) / B  # (out, in)
        if self.hebb.decay > 0.0:
            post_sq_mean = np.mean(post * post, axis=0)[:, None]  # (out,1)
            decay_term = post_sq_mean * W
        else:
            decay_term = 0.0
        return self.hebb.alpha * (hebb_outer - self.hebb.decay * decay_term)

    def _clip(self, g: Array) -> Array:
        if self.optim.clip is None or self.optim.clip <= 0:
            return g
        return np.clip(g, -self.optim.clip, self.optim.clip)

    def _step(self, dWs: List[Array], dbs: List[Array], pre_acts: List[Array], post_acts: List[Array]):
        if self.optim.use_adam:
            self._adam_t += 1
        for i, layer in enumerate(self.layers):
            gW = self._clip(dWs[i])
            gb = self._clip(dbs[i])
            # L2 weight decay
            if self.optim.weight_decay and self.optim.weight_decay > 0:
                gW = gW + self.optim.weight_decay * layer.W
            dW_hebb = self._apply_hebbian(layer.W, pre_acts[i], post_acts[i + 1])
            if self.optim.use_adam:
                layer.mW = self.optim.beta1 * layer.mW + (1 - self.optim.beta1) * gW
                layer.mb = self.optim.beta1 * layer.mb + (1 - self.optim.beta1) * gb
                layer.vW = self.optim.beta2 * layer.vW + (1 - self.optim.beta2) * (gW * gW)
                layer.vb = self.optim.beta2 * layer.vb + (1 - self.optim.beta2) * (gb * gb)
                mW_hat = layer.mW / (1 - self.optim.beta1 ** self._adam_t)
                mb_hat = layer.mb / (1 - self.optim.beta1 ** self._adam_t)
                vW_hat = layer.vW / (1 - self.optim.beta2 ** self._adam_t)
                vb_hat = layer.vb / (1 - self.optim.beta2 ** self._adam_t)
                layer.W -= self.optim.lr * mW_hat / (np.sqrt(vW_hat) + self.optim.eps)
                layer.b -= self.optim.lr * mb_hat / (np.sqrt(vb_hat) + self.optim.eps)
                layer.W += dW_hebb
            else:
                layer.vW = self.optim.momentum * layer.vW + (1 - self.optim.momentum) * gW
                layer.vb = self.optim.momentum * layer.vb + (1 - self.optim.momentum) * gb
                layer.W -= self.optim.lr * layer.vW
                layer.b -= self.optim.lr * layer.vb
                layer.W += dW_hebb

    def fit(self, X: Array, y: Array, epochs: int = 20, batch_size: int = 32, task: str = "binary",
            X_val: Optional[Array] = None, y_val: Optional[Array] = None, verbose: bool = True):
        n = X.shape[0]
        idx = np.arange(n)
        for ep in range(1, epochs + 1):
            np.random.shuffle(idx)
            total = 0.0
            for s in range(0, n, batch_size):
                bi = idx[s:s+batch_size]
                xb = X[bi]
                yb = y[bi]
                zs, ys = self.forward(xb)
                if task == "binary":
                    loss, gout = self.bce_with_logits(zs[-1], yb)
                else:
                    loss, gout = self.mse(ys[-1], yb)
                dWs, dbs = self.backward(zs, ys, gout)
                self._step(dWs, dbs, ys, ys)
                total += loss * len(bi)
            tr_loss = total / n
            if verbose:
                if X_val is not None and y_val is not None:
                    val = self.evaluate_loss(X_val, y_val, task)
                    print(f"Epoch {ep:3d} | loss {tr_loss:.4f} | val {val:.4f}")
                else:
                    print(f"Epoch {ep:3d} | loss {tr_loss:.4f}")

    def evaluate_loss(self, X: Array, y: Array, task: str = "binary") -> float:
        zs, ys = self.forward(X)
        if task == "binary":
            loss, _ = self.bce_with_logits(zs[-1], y)
        else:
            loss, _ = self.mse(ys[-1], y)
        return float(loss)

    def predict_logits(self, X: Array) -> Array:
        zs, ys = self.forward(X)
        return zs[-1]

    def predict_proba(self, X: Array) -> Array:
        return sigmoid(self.predict_logits(X))

    def predict(self, X: Array, threshold: float = 0.5) -> Array:
        return (self.predict_proba(X) >= threshold).astype(int)

    def _state_dict(self):
        sd = {}
        for i, layer in enumerate(self.layers):
            sd[f"W{i}"] = layer.W
            sd[f"b{i}"] = layer.b
        return sd

    def _load_state(self, state):
        for i, layer in enumerate(self.layers):
            layer.W[...] = state[f"W{i}"]
            layer.b[...] = state[f"b{i}"]

    def save(self, path: str):
        np.savez(path, **self._state_dict())

    def load(self, path: str):
        st = np.load(path)
        self._load_state(st)
