"""Command-line interface for the lobotoml neural compiler system."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

from .nn.model import Sequential
from .nn.layers import Dense
from .nn.quantize import QuantizationConfig, QuantizedModel, float_to_fixed, fixed_to_float
from .compiler.compiler import LobotoMLCompiler
from .compiler.ir_generator import NNToIRCompiler
from .ir.visualizer import LayerVisualizer
from .vm.vm import BrainfuckVM
from .vm.fast_vm import FastBrainfuckVM
from .experiments.benchmark import run_comprehensive_benchmark


def get_dataset(name: str):
    """Retrieve standard toy logic datasets."""
    name = name.lower().strip()
    if name == "xor":
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64)
    elif name == "and":
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        y = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float64)
    elif name == "or":
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        y = np.array([[0.0], [1.0], [1.0], [1.0]], dtype=np.float64)
    elif name == "nand":
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        y = np.array([[1.0], [1.0], [1.0], [0.0]], dtype=np.float64)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: xor, and, or, nand")
    return X, y


def cmd_train(args):
    """Train a tiny neural network and serialize it to JSON."""
    X, y = get_dataset(args.dataset)
    print(f"[*] Training {args.dataset.upper()} model (in={X.shape[1]}, hidden={args.hidden}, out={y.shape[1]})...")

    np.random.seed(args.seed)
    model = Sequential([
        Dense(X.shape[1], args.hidden, activation="relu", seed=args.seed),
        Dense(args.hidden, y.shape[1], activation="sigmoid", seed=args.seed + 100),
    ])

    loss_history = model.fit(
        X, y,
        epochs=args.epochs,
        lr=args.lr,
        loss="mse",
        optimizer="adam",
        verbose=args.verbose,
    )

    preds = model.predict(X).ravel()
    print(f"[+] Training completed. Final MSE Loss: {loss_history[-1]:.6f}")
    print("[+] Continuous Predictions on Training Set:")
    for sample_x, pred, true_y in zip(X, preds, y.ravel()):
        print(f"    in: {sample_x} -> pred: {pred:.4f} (target: {int(true_y)})")

    out_path = Path(args.output)
    model.save(out_path)
    print(f"[+] Model successfully saved to: {out_path.resolve()}")


def cmd_compile(args):
    """Compile a trained JSON model into a Brainfuck program."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[-] Error: Model file '{model_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    model = Sequential.load(model_path)
    qconfig = QuantizationConfig(scale=args.scale)
    qmodel = QuantizedModel.from_continuous_model(model, qconfig)

    compiler = LobotoMLCompiler(qconfig)
    bf_code = compiler.compile(qmodel, optimize=not args.no_optimize)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("generated") / f"{model_path.stem}.bf"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(bf_code)

    print(f"[+] Successfully compiled '{model_path}' -> '{out_path}'")
    print(f"    Brainfuck Code Size: {len(bf_code):,} bytes")
    print(f"    Quantization Scale S: {args.scale}")
    print(f"    Peephole Optimization: {'Enabled' if not args.no_optimize else 'Disabled'}")


def cmd_run(args):
    """Execute a compiled Brainfuck model on the Python Brainfuck VM."""
    bf_path = Path(args.bf_file)
    if not bf_path.exists():
        print(f"[-] Error: Brainfuck file '{bf_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(bf_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Parse inputs
    if args.raw_input:
        int_inputs = [int(v.strip()) for v in args.raw_input.split(",")]
    elif args.input:
        float_inputs = [float(v.strip()) for v in args.input.split(",")]
        int_inputs = [float_to_fixed(v, args.scale) for v in float_inputs]
    else:
        print("[-] Error: Must specify either --input (floats) or --raw-input (integers).", file=sys.stderr)
        sys.exit(1)

    if args.vm == "standard":
        vm = BrainfuckVM(code)
    else:
        vm = FastBrainfuckVM(code)

    res = vm.run(inputs=int_inputs)

    print(f"[*] Executed: {bf_path}")
    print(f"    Inputs (raw scaled): {int_inputs}")
    if args.input:
        print(f"    Inputs (float):      {[float(v.strip()) for v in args.input.split(',') ]}")
    print(f"    Outputs (raw int):   {res.outputs}")
    if res.outputs:
        float_preds = [v / float(args.scale) for v in res.outputs]
        print(f"    Outputs (float):     {float_preds}")
        binary_preds = [int(p >= 0.5) for p in float_preds]
        print(f"    Classification:      {binary_preds}")
    print(f"    VM Steps Executed:   {res.steps:,}")


def cmd_verify(args):
    """Verify bit-exact prediction equivalence between Python and Brainfuck."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[-] Error: Model file '{model_path}' not found.", file=sys.stderr)
        sys.exit(1)

    model = Sequential.load(model_path)
    X, y = get_dataset(args.dataset)
    qconfig = QuantizationConfig(scale=args.scale)
    qmodel = QuantizedModel.from_continuous_model(model, qconfig)

    compiler = LobotoMLCompiler(qconfig)
    bf_code = compiler.compile(qmodel, optimize=True)
    vm = FastBrainfuckVM(bf_code)

    print(f"[*] Verifying model '{model_path}' on '{args.dataset.upper()}' dataset (Scale S = {args.scale})...")
    print("=" * 80)
    print(f"{'Input':<15} | {'Target':<7} | {'PyFloat':<9} | {'PyFixed':<9} | {'BF Out':<9} | {'Match?':<7}")
    print("-" * 80)

    all_matched = True
    for x, target in zip(X, y.ravel()):
        py_float = model.predict(x.reshape(1, -1))[0][0]
        fixed_in = [float_to_fixed(v, args.scale) for v in x]
        py_fixed = qmodel.forward_fixed(fixed_in)[0]
        res = vm.run(inputs=fixed_in)
        bf_out = res.outputs[0] if res.outputs else 0

        match = (py_fixed == bf_out)
        if not match:
            all_matched = False

        print(f"{str(x):<15} | {int(target):<7} | {py_float:<9.4f} | {py_fixed:<9d} | {bf_out:<9d} | {'PASS' if match else 'FAIL':<7}")

    print("=" * 80)
    if all_matched:
        print("[+] VERIFICATION PASSED: Python fixed-point and Brainfuck VM predictions are 100% BIT-EXACT.")
    else:
        print("[-] VERIFICATION FAILED: Discrepancy detected between Python and Brainfuck.", file=sys.stderr)
        sys.exit(1)


def cmd_visualize(args):
    """Render ASCII visualization of the model compilation pipeline."""
    model_path = Path(args.model)
    model = Sequential.load(model_path)
    qconfig = QuantizationConfig(scale=args.scale)
    qmodel = QuantizedModel.from_continuous_model(model, qconfig)

    print(LayerVisualizer.render_model_summary(qmodel))
    for i, l in enumerate(qmodel.layers):
        print(LayerVisualizer.render_layer_compilation_diagram(
            layer_idx=i + 1,
            in_dim=l.in_features,
            out_dim=l.out_features,
            activation=l.activation_name,
            scale=args.scale,
        ))


def cmd_benchmark(args):
    """Run comprehensive performance benchmarks."""
    model_path = Path(args.model)
    model = Sequential.load(model_path)
    X, y = get_dataset(args.dataset)
    print(f"[*] Running benchmark on '{model_path}' ({args.dataset.upper()})...")
    result = run_comprehensive_benchmark(model, X, y, scale=args.scale, dataset_name=args.dataset.upper())
    print("\n" + result.summary_table())


def main():
    parser = argparse.ArgumentParser(
        prog="lobotoml",
        description="LobotoML: Neural network inference with zero frontal lobe capacity (compiled to Brainfuck).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")

    # Train
    p_train = subparsers.add_parser("train", help="Train a tiny neural network model")
    p_train.add_argument("--dataset", default="xor", choices=["xor", "and", "or", "nand"])
    p_train.add_argument("--output", "-o", default="models/xor.json", help="Path to save JSON model")
    p_train.add_argument("--hidden", type=int, default=4, help="Number of hidden neurons")
    p_train.add_argument("--epochs", type=int, default=5000, help="Training epochs")
    p_train.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    p_train.add_argument("--seed", type=int, default=0, help="Random seed for determinism")
    p_train.add_argument("--verbose", action="store_true", help="Print training loss per epoch")
    p_train.set_defaults(func=cmd_train)

    # Compile
    p_compile = subparsers.add_parser("compile", help="Compile a trained model into Brainfuck")
    p_compile.add_argument("model", help="Path to model JSON file")
    p_compile.add_argument("--output", "-o", help="Output path for .bf file")
    p_compile.add_argument("--scale", type=int, default=16, help="Fixed-point scaling factor S")
    p_compile.add_argument("--no-optimize", action="store_true", help="Disable peephole optimizer")
    p_compile.set_defaults(func=cmd_compile)

    # Run
    p_run = subparsers.add_parser("run", help="Execute compiled Brainfuck model")
    p_run.add_argument("bf_file", help="Path to .bf file")
    p_run.add_argument("--input", "-i", help="Comma-separated float inputs (e.g. 1.0,0.0)")
    p_run.add_argument("--raw-input", help="Comma-separated integer inputs (e.g. 16,0)")
    p_run.add_argument("--scale", type=int, default=16, help="Fixed-point scaling factor S")
    p_run.add_argument("--vm", default="fast", choices=["fast", "standard"], help="VM engine")
    p_run.set_defaults(func=cmd_run)

    # Verify
    p_verify = subparsers.add_parser("verify", help="Verify prediction equivalence between Python & Brainfuck")
    p_verify.add_argument("model", help="Path to model JSON file")
    p_verify.add_argument("--dataset", default="xor", choices=["xor", "and", "or", "nand"])
    p_verify.add_argument("--scale", type=int, default=16, help="Fixed-point scaling factor S")
    p_verify.set_defaults(func=cmd_verify)

    # Visualize
    p_vis = subparsers.add_parser("visualize", help="Render ASCII architecture and compilation diagrams")
    p_vis.add_argument("model", help="Path to model JSON file")
    p_vis.add_argument("--scale", type=int, default=16, help="Fixed-point scaling factor S")
    p_vis.set_defaults(func=cmd_visualize)

    # Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run comparative latency and memory benchmarks")
    p_bench.add_argument("model", help="Path to model JSON file")
    p_bench.add_argument("--dataset", default="xor", choices=["xor", "and", "or", "nand"])
    p_bench.add_argument("--scale", type=int, default=16, help="Fixed-point scaling factor S")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
