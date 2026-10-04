"""Neural network primitives and fixed-point quantization for neurofuck."""

from .activations import Activation, ReLU, Sigmoid, HardSigmoid, Linear, Step, get_activation
from .layers import Layer, Dense
from .loss import Loss, MSELoss, BinaryCrossEntropyLoss
from .model import Sequential
from .quantize import QuantizationConfig, QuantizedModel, QuantizedLayer, float_to_fixed, fixed_to_float

__all__ = [
    "Activation",
    "ReLU",
    "Sigmoid",
    "HardSigmoid",
    "Linear",
    "Step",
    "get_activation",
    "Layer",
    "Dense",
    "Loss",
    "MSELoss",
    "BinaryCrossEntropyLoss",
    "Sequential",
    "QuantizationConfig",
    "QuantizedModel",
    "QuantizedLayer",
    "float_to_fixed",
    "fixed_to_float",
]
