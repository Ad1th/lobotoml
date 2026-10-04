"""Neurofuck: A tiny neural-network inference system compiled into Brainfuck."""

from .nn import (
    Sequential,
    Dense,
    ReLU,
    Sigmoid,
    HardSigmoid,
    Linear,
    Step,
    QuantizationConfig,
    QuantizedModel,
    float_to_fixed,
    fixed_to_float,
)
from .compiler import NeurofuckCompiler, TapeMemoryManager, BrainfuckOptimizer
from .vm import BrainfuckVM, FastBrainfuckVM, VMConfig, VMResult
from .ir import IRProgram, LayerVisualizer

__version__ = "0.1.0"
__all__ = [
    "Sequential",
    "Dense",
    "ReLU",
    "Sigmoid",
    "HardSigmoid",
    "Linear",
    "Step",
    "QuantizationConfig",
    "QuantizedModel",
    "float_to_fixed",
    "fixed_to_float",
    "NeurofuckCompiler",
    "TapeMemoryManager",
    "BrainfuckOptimizer",
    "BrainfuckVM",
    "FastBrainfuckVM",
    "VMConfig",
    "VMResult",
    "IRProgram",
    "LayerVisualizer",
]
