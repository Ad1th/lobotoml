"""Neural network layers."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from .activations import Activation, get_activation, Linear


class Layer(ABC):
    """Abstract base class for all neural layers."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Layer":
        pass


class Dense(Layer):
    """Fully-connected (Dense) feed-forward layer: y = activation(x @ W + b)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[str | Activation] = None,
        use_bias: bool = True,
        seed: Optional[int] = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.activation: Activation = get_activation(activation)
        self.use_bias = use_bias

        rng = np.random.RandomState(seed)
        # Xavier / Glorot uniform initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weights: np.ndarray = rng.uniform(-limit, limit, (in_features, out_features)).astype(np.float64)
        
        if self.use_bias:
            self.biases: np.ndarray = np.zeros(out_features, dtype=np.float64)
        else:
            self.biases = np.zeros(out_features, dtype=np.float64)

        # Caches for backpropagation
        self._last_input: Optional[np.ndarray] = None
        self._last_linear_output: Optional[np.ndarray] = None
        self.grad_weights: Optional[np.ndarray] = None
        self.grad_biases: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            x: Array of shape (batch_size, in_features) or (in_features,)
        Returns:
            Array of shape (batch_size, out_features) or (out_features,)
        """
        is_1d = x.ndim == 1
        if is_1d:
            x_2d = x.reshape(1, -1)
        else:
            x_2d = x

        self._last_input = x_2d
        linear_output = np.matmul(x_2d, self.weights)
        if self.use_bias:
            linear_output = linear_output + self.biases
        self._last_linear_output = linear_output

        activated = self.activation.forward(linear_output)
        return activated[0] if is_1d else activated

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass computing gradients and backpropagating loss gradient.
        
        Args:
            grad_output: Upstream gradient dL/dY of shape (batch_size, out_features) or (out_features,)
        Returns:
            Downstream gradient dL/dX of shape matching input.
        """
        is_1d = grad_output.ndim == 1
        if is_1d:
            grad_output_2d = grad_output.reshape(1, -1)
        else:
            grad_output_2d = grad_output

        # Derivative through activation function
        d_linear = self.activation.backward(self._last_linear_output, grad_output_2d)

        # Gradients w.r.t parameters
        self.grad_weights = np.matmul(self._last_input.T, d_linear)
        if self.use_bias:
            self.grad_biases = np.sum(d_linear, axis=0)
        else:
            self.grad_biases = np.zeros_like(self.biases)

        # Gradient w.r.t input
        grad_input = np.matmul(d_linear, self.weights.T)
        return grad_input[0] if is_1d else grad_input

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Dense",
            "in_features": int(self.in_features),
            "out_features": int(self.out_features),
            "activation": self.activation.name,
            "use_bias": bool(self.use_bias),
            "weights": self.weights.tolist(),
            "biases": self.biases.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dense":
        layer = cls(
            in_features=data["in_features"],
            out_features=data["out_features"],
            activation=data.get("activation", "linear"),
            use_bias=data.get("use_bias", True),
        )
        layer.weights = np.array(data["weights"], dtype=np.float64)
        layer.biases = np.array(data["biases"], dtype=np.float64)
        return layer
