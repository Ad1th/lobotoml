"""Fixed-point quantization for neural networks."""

from dataclasses import dataclass
from typing import List, Dict, Any, Union
import numpy as np

from .activations import Activation, get_activation
from .layers import Dense
from .model import Sequential


@dataclass
class QuantizationConfig:
    """Configuration for fixed-point quantization.
    
    Attributes:
        scale: Integer scale factor S (e.g. 64 for Q6 fixed-point or 100 for decimal).
               A float value x is represented as round(x * scale).
        bits: Bit-width of the target integer representation (e.g. 16 or 32).
        round_mode: Rounding mode ('round', 'floor', 'trunc').
    """
    scale: int = 64
    bits: int = 16
    round_mode: str = "round"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale": self.scale,
            "bits": self.bits,
            "round_mode": self.round_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizationConfig":
        return cls(
            scale=data.get("scale", 64),
            bits=data.get("bits", 16),
            round_mode=data.get("round_mode", "round"),
        )


def float_to_fixed(val: Union[float, int, list, np.ndarray], scale: int = 64) -> Union[int, np.ndarray]:
    """Convert float value(s) to scaled fixed-point integer(s)."""
    if isinstance(val, (list, tuple)):
        return [int(round(float(v) * scale)) for v in val]
    if isinstance(val, np.ndarray):
        return np.round(val * scale).astype(np.int64)
    return int(round(float(val) * scale))


def fixed_to_float(val: Union[int, np.ndarray], scale: int = 64) -> Union[float, np.ndarray]:
    """Convert fixed-point integer(s) back to float value(s)."""
    if isinstance(val, np.ndarray):
        return val.astype(np.float64) / float(scale)
    return float(val) / float(scale)


class QuantizedLayer:
    """A neural network layer with quantized integer weights and biases."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weights: List[List[int]],
        biases: List[int],
        activation: str = "linear",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = weights  # 2D list of shape [in_features][out_features]
        self.biases = biases    # 1D list of length out_features
        self.activation_name = activation
        self.activation: Activation = get_activation(activation)

    def forward_fixed(self, inputs: List[int], scale: int) -> List[int]:
        """Perform bit-exact fixed-point forward pass for a single input vector.
        
        Args:
            inputs: List of integer fixed-point inputs (length in_features).
            scale: Fixed-point scaling factor.
        Returns:
            List of integer fixed-point outputs (length out_features).
        """
        assert len(inputs) == self.in_features, f"Expected {self.in_features} inputs, got {len(inputs)}"
        outputs: List[int] = []

        for j in range(self.out_features):
            # Accumulate dot product: sum(input_i * weight_ij)
            # Each product is scaled by S^2
            acc = 0
            for i in range(self.in_features):
                acc += inputs[i] * self.weights[i][j]

            # Rescale back to scale S: acc // scale (using trunc/floor division)
            # In Brainfuck integer arithmetic, division is integer division (truncating towards zero)
            if acc >= 0:
                rescaled = acc // scale
            else:
                rescaled = -((-acc) // scale)

            # Add bias (already at scale S)
            pre_activation = rescaled + self.biases[j]

            # Apply activation function in fixed-point
            activated = self.activation.forward_fixed(pre_activation, scale)
            outputs.append(activated)

        return outputs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "QuantizedDense",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "activation": self.activation_name,
            "weights": self.weights,
            "biases": self.biases,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizedLayer":
        return cls(
            in_features=data["in_features"],
            out_features=data["out_features"],
            weights=data["weights"],
            biases=data["biases"],
            activation=data.get("activation", "linear"),
        )


class QuantizedModel:
    """A quantized neural network composed of integer-arithmetic layers."""

    def __init__(self, layers: List[QuantizedLayer], config: QuantizationConfig):
        self.layers = layers
        self.config = config

    @property
    def scale(self) -> int:
        return self.config.scale

    def forward_fixed(self, inputs: List[int]) -> List[int]:
        """Bit-exact forward pass on integer fixed-point vector."""
        current = inputs
        for layer in self.layers:
            current = layer.forward_fixed(current, self.config.scale)
        return current

    def forward_float(self, inputs: List[float]) -> List[float]:
        """Helper forward pass accepting and returning floating point values."""
        fixed_inputs = [float_to_fixed(x, self.config.scale) for x in inputs]
        fixed_outputs = self.forward_fixed(fixed_inputs)
        return [fixed_to_float(y, self.config.scale) for y in fixed_outputs]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "QuantizedSequential",
            "quantization": self.config.to_dict(),
            "layers": [l.to_dict() for l in self.layers],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizedModel":
        config = QuantizationConfig.from_dict(data.get("quantization", {}))
        layers = [QuantizedLayer.from_dict(ld) for ld in data["layers"]]
        return cls(layers=layers, config=config)

    @classmethod
    def from_continuous_model(
        cls,
        model: Sequential,
        config: Optional[QuantizationConfig] = None,
    ) -> "QuantizedModel":
        """Quantize a continuous float Sequential model into fixed-point integer model."""
        if config is None:
            config = QuantizationConfig()

        q_layers: List[QuantizedLayer] = []
        for layer in model.layers:
            if isinstance(layer, Dense):
                # Quantize weights and biases
                w_fixed = float_to_fixed(layer.weights, config.scale).tolist()
                b_fixed = float_to_fixed(layer.biases, config.scale).tolist()
                q_layers.append(
                    QuantizedLayer(
                        in_features=layer.in_features,
                        out_features=layer.out_features,
                        weights=w_fixed,
                        biases=b_fixed,
                        activation=layer.activation.name,
                    )
                )
            else:
                raise ValueError(f"Cannot quantize layer of type {type(layer)}")

        return cls(layers=q_layers, config=config)
