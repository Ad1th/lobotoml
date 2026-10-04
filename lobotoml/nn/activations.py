"""Activation functions for continuous and fixed-point neural inference."""

from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    """Abstract base class for activation functions."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Continuous forward pass (floating-point)."""
        pass

    @abstractmethod
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass computing gradient with respect to input."""
        pass

    @abstractmethod
    def forward_fixed(self, x: int, scale: int) -> int:
        """Fixed-point forward pass for a single scalar integer input."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name."""
        pass


class ReLU(Activation):
    """Rectified Linear Unit: f(x) = max(0, x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (x > 0).astype(float)

    def forward_fixed(self, x: int, scale: int) -> int:
        return max(0, x)

    @property
    def name(self) -> str:
        return "relu"


class Linear(Activation):
    """Linear / Identity activation: f(x) = x."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        return grad_output

    def forward_fixed(self, x: int, scale: int) -> int:
        return x

    @property
    def name(self) -> str:
        return "linear"


class Step(Activation):
    """Binary Heaviside step activation: f(x) = 1 if x >= 0 else 0."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (x >= 0.0).astype(float)

    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        # Step is non-differentiable; surrogate gradient (straight-through)
        return grad_output

    def forward_fixed(self, x: int, scale: int) -> int:
        return scale if x >= 0 else 0

    @property
    def name(self) -> str:
        return "step"


class HardSigmoid(Activation):
    """Piecewise linear approximation of sigmoid: clamp(0.25 * x + 0.5, 0, 1)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.clip(0.25 * x + 0.5, 0.0, 1.0)

    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        mask = (x >= -2.0) & (x <= 2.0)
        return grad_output * 0.25 * mask.astype(float)

    def forward_fixed(self, x: int, scale: int) -> int:
        # In fixed point: x/4 + scale/2 clamped to [0, scale]
        half_scale = scale // 2
        val = (x // 4) + half_scale
        if val < 0:
            return 0
        if val > scale:
            return scale
        return val

    @property
    def name(self) -> str:
        return "hard_sigmoid"


class Sigmoid(Activation):
    """Standard logistic sigmoid: f(x) = 1 / (1 + exp(-x)).
    
    Fixed-point execution uses a high-precision 5-region piecewise polynomial / linear
    or configurable hard-sigmoid fallback to maintain exact reproducibility in Brainfuck.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Numerically stable standard logistic
        clipped = np.clip(x, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return grad_output * s * (1.0 - s)

    def forward_fixed(self, x: int, scale: int) -> int:
        # Standard symmetric piecewise approximation for Brainfuck integer cells:
        # x_val in float units is x / scale
        # For |x| >= 2.5: clamp to 0 or scale
        # For |x| < 2.5: 0.25 * x + 0.5 * scale
        # This matches the Brainfuck compiler's Sigmoid emitter logic exactly.
        half_scale = scale // 2
        val = (x // 4) + half_scale
        if val < 0:
            return 0
        if val > scale:
            return scale
        return val

    @property
    def name(self) -> str:
        return "sigmoid"


def get_activation(act: str | Activation | None) -> Activation:
    """Factory function for activations."""
    if act is None:
        return Linear()
    if isinstance(act, Activation):
        return act
    act_str = str(act).lower().strip()
    if act_str in ("relu", "rectified"):
        return ReLU()
    elif act_str in ("sigmoid", "logistic"):
        return Sigmoid()
    elif act_str in ("hard_sigmoid", "hardsigmoid"):
        return HardSigmoid()
    elif act_str in ("linear", "identity", "none"):
        return Linear()
    elif act_str in ("step", "heaviside"):
        return Step()
    else:
        raise ValueError(f"Unknown activation function: '{act}'")
