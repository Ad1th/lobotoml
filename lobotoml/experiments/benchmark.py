"""Benchmark comparing Python float, Python fixed-point, and Brainfuck VM execution."""

import time
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import numpy as np

from ..nn.model import Sequential
from ..nn.quantize import QuantizedModel, QuantizationConfig, float_to_fixed, fixed_to_float
from ..compiler.compiler import LobotoMLCompiler
from ..vm.fast_vm import FastBrainfuckVM
from ..vm.vm import BrainfuckVM


@dataclass
class BenchmarkResult:
    """Structured performance and accuracy metrics."""
    dataset_name: str
    num_samples: int
    scale: int
    # Predictions and Accuracies
    py_float_accuracy: float
    py_fixed_accuracy: float
    bf_accuracy: float
    exact_equivalence_rate: float
    # Latencies (milliseconds per sample)
    py_float_latency_ms: float
    py_fixed_latency_ms: float
    fast_bf_latency_ms: float
    standard_bf_latency_ms: float
    # Code and Memory Metrics
    bf_code_length_bytes: int
    model_param_count: int
    avg_bf_steps: float
    max_bf_tape_cells: int

    def summary_table(self) -> str:
        """Format metrics into a clean markdown table."""
        lines = [
            f"### Benchmark Report: {self.dataset_name} (Scale S = {self.scale})",
            "",
            "| Metric | Python Float (Float64) | Python Quantized (Fixed-Point) | Brainfuck VM (Compiled) |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Accuracy** | {self.py_float_accuracy * 100:.1f}% | {self.py_fixed_accuracy * 100:.1f}% | {self.bf_accuracy * 100:.1f}% |",
            f"| **Latency / Sample** | {self.py_float_latency_ms * 1000:.2f} µs | {self.py_fixed_latency_ms * 1000:.2f} µs | {self.fast_bf_latency_ms:.2f} ms |",
            f"| **Equivalence with PyFixed** | N/A | 100.0% | {self.exact_equivalence_rate * 100:.1f}% bit-exact |",
            f"| **Representation Size** | {self.model_param_count * 8} bytes (raw float) | {self.model_param_count * 2} bytes (int16) | {self.bf_code_length_bytes} bytes (.bf code) |",
            f"| **Average VM Steps** | N/A | N/A | {self.avg_bf_steps:,.0f} instructions |",
            f"| **Max Tape Cell Used** | N/A | N/A | Cell {self.max_bf_tape_cells} |",
            "",
            f"> **Equivalence Rate:** `{self.exact_equivalence_rate * 100:.1f}%` bit-exact match across all test cases.",
        ]
        return "\n".join(lines)


def run_comprehensive_benchmark(
    model: Sequential,
    X: np.ndarray,
    y: np.ndarray,
    scale: int = 16,
    dataset_name: str = "XOR",
    runs: int = 5,
) -> BenchmarkResult:
    """Run rigorous comparative benchmarks between floating point, fixed point, and Brainfuck."""
    num_samples = len(X)
    param_count = sum(l.weights.size + l.biases.size for l in model.layers if hasattr(l, "weights"))

    qconfig = QuantizationConfig(scale=scale)
    qmodel = QuantizedModel.from_continuous_model(model, qconfig)

    compiler = LobotoMLCompiler(qconfig)
    bf_code = compiler.compile(qmodel, optimize=True)

    vm = FastBrainfuckVM(bf_code)
    std_vm = BrainfuckVM(bf_code)

    # 1. Measure Python Float
    start = time.perf_counter()
    for _ in range(runs):
        py_float_preds = model.predict(X)
    py_float_time = (time.perf_counter() - start) / (runs * num_samples) * 1000.0

    # 2. Measure Python Fixed
    start = time.perf_counter()
    for _ in range(runs):
        py_fixed_preds = []
        for x in X:
            fin = [float_to_fixed(v, scale) for v in x]
            py_fixed_preds.append(qmodel.forward_fixed(fin))
    py_fixed_time = (time.perf_counter() - start) / (runs * num_samples) * 1000.0

    # 3. Measure Fast Brainfuck VM
    bf_outputs = []
    bf_steps_list = []
    max_tape_used = 0

    start = time.perf_counter()
    for x in X:
        fin = [float_to_fixed(v, scale) for v in x]
        res = vm.run(inputs=fin)
        bf_outputs.append(res.outputs)
        bf_steps_list.append(res.steps)
        if res.tape_snapshot:
            max_tape_used = max(max_tape_used, max(res.tape_snapshot.keys()))
    fast_bf_time = (time.perf_counter() - start) / num_samples * 1000.0

    # 4. Measure Standard Brainfuck VM on a subset
    start = time.perf_counter()
    fin0 = [float_to_fixed(v, scale) for v in X[0]]
    std_vm.run(inputs=fin0)
    std_bf_time = (time.perf_counter() - start) * 1000.0

    # Accuracies
    y_true = y.ravel()
    py_float_binary = (py_float_preds.ravel() >= 0.5).astype(int)
    py_float_acc = float(np.mean(py_float_binary == y_true))

    py_fixed_binary = [int(p[0] >= (scale // 2)) for p in py_fixed_preds]
    py_fixed_acc = float(np.mean(np.array(py_fixed_binary) == y_true))

    bf_binary = [int(p[0] >= (scale // 2)) for p in bf_outputs]
    bf_acc = float(np.mean(np.array(bf_binary) == y_true))

    # Equivalence check
    matches = sum(1 for p, b in zip(py_fixed_preds, bf_outputs) if p == b)
    equiv_rate = float(matches / num_samples)

    return BenchmarkResult(
        dataset_name=dataset_name,
        num_samples=num_samples,
        scale=scale,
        py_float_accuracy=py_float_acc,
        py_fixed_accuracy=py_fixed_acc,
        bf_accuracy=bf_acc,
        exact_equivalence_rate=equiv_rate,
        py_float_latency_ms=py_float_time,
        py_fixed_latency_ms=py_fixed_time,
        fast_bf_latency_ms=fast_bf_time,
        standard_bf_latency_ms=std_bf_time,
        bf_code_length_bytes=len(bf_code.encode("utf-8")),
        model_param_count=param_count,
        avg_bf_steps=float(np.mean(bf_steps_list)),
        max_bf_tape_cells=max_tape_used,
    )
