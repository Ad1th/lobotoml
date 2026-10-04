"""Translates neural networks into intermediate representation (NIR)."""

from typing import List, Optional
from ..nn.model import Sequential
from ..nn.quantize import QuantizedModel, QuantizedLayer, QuantizationConfig
from ..ir.graph import IRProgram
from ..ir.instructions import (
    Alloc,
    Free,
    SetConst,
    AddConst,
    SubConst,
    Move,
    Copy,
    Add,
    Sub,
    Mul,
    MulConst,
    DivConst,
    ReLU,
    Sigmoid,
    ReadInput,
    PrintOutput,
    Comment,
)


class NNToIRCompiler:
    """Compiles a QuantizedModel into a linear NIR instruction stream."""

    def __init__(self, model: QuantizedModel):
        self.model = model
        self.scale = model.scale

    def compile(self) -> IRProgram:
        """Generate NIR program from quantized model."""
        prog = IRProgram()
        prog.comment("=== NEUROFUCK NEURAL NETWORK INTERMEDIATE REPRESENTATION ===")
        prog.comment(f"Fixed-point scale factor S = {self.scale}")

        first_layer = self.model.layers[0]
        in_dim = first_layer.in_features

        # Allocate input registers
        prog.comment("--- Layer 0: Input Registers ---")
        for i in range(in_dim):
            prog.emit(Alloc(name=f"in_{i}", initial_value=0))
            prog.emit(ReadInput(target=f"in_{i}", index=i))

        current_layer_inputs = [f"in_{i}" for i in range(in_dim)]

        # Compile each Dense layer
        for l_idx, layer in enumerate(self.model.layers):
            prog.comment(f"--- Layer {l_idx + 1}: Dense ({layer.in_features} -> {layer.out_features}, act={layer.activation_name}) ---")
            layer_outputs = []

            for j in range(layer.out_features):
                acc_var = f"l{l_idx}_acc_{j}"
                out_var = f"l{l_idx}_out_{j}"
                prog.emit(Alloc(name=acc_var, initial_value=0))
                prog.emit(Alloc(name=out_var, initial_value=0))

                # Multiply-accumulate: sum(x_i * W_ij)
                for i in range(layer.in_features):
                    w = layer.weights[i][j]
                    if w != 0:
                        term_var = f"l{l_idx}_t_{i}_{j}"
                        prog.emit(Alloc(name=term_var, initial_value=0))
                        prog.emit(MulConst(target=term_var, source=current_layer_inputs[i], const=w))
                        prog.emit(Add(target=acc_var, source=term_var))
                        prog.emit(Free(name=term_var))

                # Integer division by scale factor S
                rescaled_var = f"l{l_idx}_rescaled_{j}"
                prog.emit(Alloc(name=rescaled_var, initial_value=0))
                prog.emit(DivConst(target=rescaled_var, source=acc_var, divisor=self.scale))

                # Add bias
                b = layer.biases[j]
                if b != 0:
                    prog.emit(AddConst(target=rescaled_var, value=b))

                # Activation
                act = layer.activation_name.lower()
                if act == "relu":
                    prog.emit(ReLU(target=out_var, source=rescaled_var))
                elif act in ("sigmoid", "hard_sigmoid"):
                    prog.emit(Sigmoid(target=out_var, source=rescaled_var, scale=self.scale))
                elif act in ("linear", "none"):
                    prog.emit(Move(target=out_var, source=rescaled_var))
                else:
                    raise NotImplementedError(f"Unsupported activation in IR compiler: {act}")

                prog.emit(Free(name=rescaled_var))
                prog.emit(Free(name=acc_var))
                layer_outputs.append(out_var)

            # Free previous layer input registers if not original inputs
            if l_idx > 0:
                for var in current_layer_inputs:
                    prog.emit(Free(name=var))

            current_layer_inputs = layer_outputs

        # Output final predictions
        prog.comment("--- Network Output ---")
        for idx, out_var in enumerate(current_layer_inputs):
            prog.emit(PrintOutput(source=out_var, index=idx))

        return prog
