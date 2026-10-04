"""End-to-end compiler translating neural networks to Brainfuck programs."""

from typing import List, Optional, Union
from ..nn.model import Sequential
from ..nn.quantize import QuantizedModel, QuantizationConfig
from .memory import TapeMemoryManager
from .bf_emitter import BrainfuckEmitter
from .optimizer import BrainfuckOptimizer
from .ir_generator import NNToIRCompiler


class LobotoMLCompiler:
    """Deterministic end-to-end Neural Network to Brainfuck compiler."""

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

    def compile(
        self,
        model: Union[Sequential, QuantizedModel],
        embedded_inputs: Optional[List[int]] = None,
        optimize: bool = True,
    ) -> str:
        """Compile a neural network into executable Brainfuck code.
        
        Args:
            model: Sequential float model or QuantizedModel.
            embedded_inputs: If provided, embeds constant input values at startup.
                             If None, emits ',' instructions to read inputs at runtime.
            optimize: If True, runs peephole optimizer passes.
        Returns:
            Brainfuck source code string.
        """
        if isinstance(model, Sequential):
            qmodel = QuantizedModel.from_continuous_model(model, self.config)
        else:
            qmodel = model

        scale = qmodel.scale
        mem = TapeMemoryManager()
        emitter = BrainfuckEmitter(mem)

        # Allocate scratch cells for general arithmetic
        scratch = [mem.allocate_scratch(f"scratch_{i}") for i in range(8)]
        s0, s1, s2, s3, s4, s5, s6, s7 = scratch

        # Allocate input registers (dual-rail)
        in_dim = qmodel.layers[0].in_features
        input_vars = []
        for i in range(in_dim):
            pos_v = f"in_{i}_pos"
            neg_v = f"in_{i}_neg"
            mem.allocate(pos_v)
            mem.allocate(neg_v)
            input_vars.append((pos_v, neg_v))

        # Initialize inputs
        if embedded_inputs is not None:
            for i, val in enumerate(embedded_inputs):
                pos_v, neg_v = input_vars[i]
                if val >= 0:
                    emitter.set_const(pos_v, val)
                    emitter.zero(neg_v)
                else:
                    emitter.zero(pos_v)
                    emitter.set_const(neg_v, -val)
        else:
            for i in range(in_dim):
                pos_v, neg_v = input_vars[i]
                emitter.move_to(pos_v)
                emitter.emit_raw(",")
                emitter.zero(neg_v)

        current_inputs = input_vars

        # Compile each layer
        for l_idx, layer in enumerate(qmodel.layers):
            layer_outputs = []

            for j in range(layer.out_features):
                acc_pos = f"l{l_idx}_n{j}_acc_pos"
                acc_neg = f"l{l_idx}_n{j}_acc_neg"
                mem.allocate(acc_pos)
                mem.allocate(acc_neg)
                emitter.zero(acc_pos)
                emitter.zero(acc_neg)

                # 1. Multiply-Accumulate: sum(x_i * W_ij)
                for i in range(layer.in_features):
                    w = layer.weights[i][j]
                    if w == 0:
                        continue

                    in_p, in_n = current_inputs[i]
                    abs_w = abs(w)

                    if w > 0:
                        # acc_pos += in_pos * w ; acc_neg += in_neg * w
                        emitter.multiply_add_const(in_p, acc_pos, abs_w, s0)
                        emitter.multiply_add_const(in_n, acc_neg, abs_w, s0)
                    else:
                        # acc_pos += in_neg * |w| ; acc_neg += in_pos * |w|
                        emitter.multiply_add_const(in_n, acc_pos, abs_w, s0)
                        emitter.multiply_add_const(in_p, acc_neg, abs_w, s0)

                # 2. Dual-rail normalize accumulator
                emitter.normalize_dual_rail(acc_pos, acc_neg, s0, s1, s2, s3)

                # 3. Rescale division by S
                # Division consumes acc_pos / acc_neg into quotients
                q_pos = f"l{l_idx}_n{j}_q_pos"
                q_neg = f"l{l_idx}_n{j}_q_neg"
                mem.allocate(q_pos)
                mem.allocate(q_neg)

                emitter.divide_const(acc_pos, q_pos, scale, s0, s1, s2)
                emitter.divide_const(acc_neg, q_neg, scale, s0, s1, s2)

                # 4. Add bias to rescaled pre-activation
                b = layer.biases[j]
                if b > 0:
                    emitter.add_const(q_pos, b)
                elif b < 0:
                    emitter.add_const(q_neg, -b)

                # Normalize again after bias addition
                emitter.normalize_dual_rail(q_pos, q_neg, s0, s1, s2, s3)

                # 5. Non-linear Activation
                out_pos = f"l{l_idx}_n{j}_out_pos"
                out_neg = f"l{l_idx}_n{j}_out_neg"
                mem.allocate(out_pos)
                mem.allocate(out_neg)
                emitter.zero(out_pos)
                emitter.zero(out_neg)

                act = layer.activation_name.lower()
                if act == "relu":
                    # ReLU: keep positive part, discard negative part
                    emitter.zero(q_neg)
                    emitter.move_cell(q_pos, out_pos)
                    emitter.zero(out_neg)
                elif act in ("sigmoid", "hard_sigmoid"):
                    # Hard-sigmoid activation
                    emitter.hard_sigmoid(q_pos, q_neg, out_pos, scale, scratch)
                    emitter.zero(out_neg)
                elif act in ("linear", "none"):
                    emitter.move_cell(q_pos, out_pos)
                    emitter.move_cell(q_neg, out_neg)
                else:
                    raise NotImplementedError(f"Unsupported activation in Brainfuck emitter: {act}")

                # Clean up temporary registers
                mem.free(acc_pos)
                mem.free(acc_neg)
                mem.free(q_pos)
                mem.free(q_neg)

                layer_outputs.append((out_pos, out_neg))

            current_inputs = layer_outputs

        # Output predictions via '.' instruction
        for out_pos, _ in current_inputs:
            emitter.move_to(out_pos)
            emitter.emit_raw(".")

        raw_bf = emitter.get_code()

        if optimize:
            return BrainfuckOptimizer.optimize(raw_bf)
        return raw_bf


# Backward-compatibility alias
NeurofuckCompiler = LobotoMLCompiler
