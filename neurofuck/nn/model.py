"""Sequential neural network model with training and serialization."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

from .layers import Layer, Dense
from .loss import Loss, MSELoss, BinaryCrossEntropyLoss


class Sequential:
    """Sequential Feed-Forward Neural Network container."""

    def __init__(self, layers: Optional[List[Layer]] = None):
        self.layers: List[Layer] = []
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer: Layer) -> "Sequential":
        """Add a layer to the network."""
        self.layers.append(layer)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def backward(self, loss_grad: np.ndarray) -> np.ndarray:
        """Backward pass propagating gradients from output to input."""
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 2000,
        lr: float = 0.05,
        loss: Union[str, Loss] = "mse",
        optimizer: str = "adam",
        verbose: bool = False,
        log_interval: int = 500,
    ) -> List[float]:
        """Train model using full-batch or mini-batch gradient descent / Adam.
        
        Args:
            X: Training inputs (N, in_features)
            y: Training targets (N, out_features)
            epochs: Number of training epochs
            lr: Learning rate
            loss: Loss function ('mse' or 'bce' or Loss instance)
            optimizer: 'adam' or 'sgd'
            verbose: If True, prints progress
            log_interval: Interval for printing loss
        Returns:
            List of loss values per epoch
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        loss_fn: Loss
        if isinstance(loss, Loss):
            loss_fn = loss
        elif str(loss).lower() in ("bce", "binary_cross_entropy"):
            loss_fn = BinaryCrossEntropyLoss()
        else:
            loss_fn = MSELoss()

        # Adam optimizer state
        m_w: Dict[int, np.ndarray] = {}
        v_w: Dict[int, np.ndarray] = {}
        m_b: Dict[int, np.ndarray] = {}
        v_b: Dict[int, np.ndarray] = {}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                m_w[i] = np.zeros_like(layer.weights)
                v_w[i] = np.zeros_like(layer.weights)
                m_b[i] = np.zeros_like(layer.biases)
                v_b[i] = np.zeros_like(layer.biases)

        loss_history: List[float] = []

        for epoch in range(1, epochs + 1):
            # Forward pass
            y_pred = self.forward(X)
            current_loss = loss_fn.forward(y_pred, y)
            loss_history.append(current_loss)

            # Backward pass
            loss_grad = loss_fn.backward(y_pred, y)
            self.backward(loss_grad)

            # Parameter update
            t += 1
            for i, layer in enumerate(self.layers):
                if isinstance(layer, Dense):
                    gw = layer.grad_weights
                    gb = layer.grad_biases

                    if optimizer.lower() == "adam":
                        # Weight update
                        m_w[i] = beta1 * m_w[i] + (1 - beta1) * gw
                        v_w[i] = beta2 * v_w[i] + (1 - beta2) * (gw ** 2)
                        m_w_hat = m_w[i] / (1 - beta1 ** t)
                        v_w_hat = v_w[i] / (1 - beta2 ** t)
                        layer.weights -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

                        # Bias update
                        if layer.use_bias:
                            m_b[i] = beta1 * m_b[i] + (1 - beta1) * gb
                            v_b[i] = beta2 * v_b[i] + (1 - beta2) * (gb ** 2)
                            m_b_hat = m_b[i] / (1 - beta1 ** t)
                            v_b_hat = v_b[i] / (1 - beta2 ** t)
                            layer.biases -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
                    else:
                        # Standard SGD
                        layer.weights -= lr * gw
                        if layer.use_bias:
                            layer.biases -= lr * gb

            if verbose and (epoch % log_interval == 0 or epoch == epochs):
                print(f"Epoch {epoch:5d}/{epochs} | Loss: {current_loss:.6f}")

        return loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Continuous model prediction."""
        return self.forward(X)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model architecture and weights to a dictionary."""
        return {
            "model_type": "Sequential",
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sequential":
        """Deserialize model from a dictionary."""
        model = cls()
        for layer_data in data["layers"]:
            layer_type = layer_data.get("type", "Dense")
            if layer_type == "Dense":
                model.add(Dense.from_dict(layer_data))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
        return model

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Sequential":
        """Load model from a JSON file."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
