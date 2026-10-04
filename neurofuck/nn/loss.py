"""Loss functions for training neural networks."""

from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass


class MSELoss(Loss):
    """Mean Squared Error Loss: L = 0.5 * mean((y_pred - y_true)^2)."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(0.5 * np.mean((y_pred - y_true) ** 2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n = y_pred.size
        return (y_pred - y_true) / max(1, n)


class BinaryCrossEntropyLoss(Loss):
    """Binary Cross Entropy Loss: L = -mean(y_true * log(p) + (1 - y_true) * log(1 - p))."""

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        p = np.clip(y_pred, self.eps, 1.0 - self.eps)
        loss = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        return float(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, self.eps, 1.0 - self.eps)
        grad = (p - y_true) / (p * (1.0 - p) + self.eps)
        n = y_pred.size
        return grad / max(1, n)
