"""ASCII and Unicode visualizers for neural network compilation to Brainfuck."""

from typing import Dict, List, Any, Optional
from ..nn.quantize import QuantizedModel, QuantizedLayer


def format_tape_layout(tape_map: Dict[str, int], tape_size: int = 30) -> str:
    """Render a visual ASCII representation of the Brainfuck memory tape."""
    # Invert mapping to cell -> symbol
    cell_to_sym: Dict[int, List[str]] = {}
    for sym, idx in tape_map.items():
        cell_to_sym.setdefault(idx, []).append(sym)

    max_cell = max(tape_map.values(), default=0)
    display_size = max(max_cell + 1, min(tape_size, 20))

    header = "Memory Tape Layout:"
    lines = [header, "┌" + "───────┬" * (display_size - 1) + "───────┐"]

    # Cell indices
    index_row = "│"
    for i in range(display_size):
        index_row += f" c{i:02d}  │"
    lines.append(index_row)
    lines.append("├" + "───────┼" * (display_size - 1) + "───────┤")

    # Symbols row
    sym_row = "│"
    for i in range(display_size):
        syms = cell_to_sym.get(i, ["·"])
        label = syms[0][:5]
        sym_row += f" {label:<5}│"
    lines.append(sym_row)
    lines.append("└" + "───────┴" * (display_size - 1) + "───────┘")

    return "\n".join(lines)


class LayerVisualizer:
    """ASCII diagrams displaying the compilation of Neural Layers into Brainfuck."""

    @staticmethod
    def render_model_summary(model: QuantizedModel) -> str:
        """Render ASCII computational graph for the quantized model."""
        lines = [
            "================================================================================",
            "                    LOBOTOML COMPILED NEURAL NETWORK GRAPH                      ",
            f"                     (Fixed-Point Scale Factor S = {model.scale})                ",
            "================================================================================",
        ]

        for idx, layer in enumerate(model.layers):
            lines.append(f"\n[ Layer {idx + 1}: QuantizedDense ]")
            lines.append(f"  Shape: ({layer.in_features} -> {layer.out_features}) | Activation: {layer.activation_name.upper()}")
            lines.append("  Mathematical Formulation:")
            lines.append(f"    1. Linear Accumulation :  acc_j = ∑ (x_i * W_ij)    [Scale: S²]")
            lines.append(f"    2. Integer Rescaling   :  z_j   = (acc_j // {model.scale}) + b_j  [Scale: S]")
            lines.append(f"    3. Non-linear Action   :  y_j   = {layer.activation_name.upper()}(z_j)")
            lines.append("\n  Weight Matrix & Biases:")

            for i in range(layer.in_features):
                row_str = "    in[" + str(i) + "] ───> "
                for j in range(layer.out_features):
                    w = layer.weights[i][j]
                    row_str += f"[ W_{i}{j}={w:4d} ]───> "
                lines.append(row_str + f"acc")
            
            bias_str = "    biases:      "
            for j in range(layer.out_features):
                b = layer.biases[j]
                bias_str += f"[ b_{j}={b:4d} ]      "
            lines.append(bias_str)

        lines.append("\n================================================================================")
        return "\n".join(lines)

    @staticmethod
    def render_layer_compilation_diagram(
        layer_idx: int,
        in_dim: int,
        out_dim: int,
        activation: str,
        scale: int,
    ) -> str:
        """Render detailed compilation walkthrough for a single layer."""
        diagram = f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER {layer_idx}: DENSE ({in_dim} -> {out_dim}, {activation.upper()}) COMPILATION PIPELINE                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Neural Domain]                 [NIR Domain]           [Brainfuck Memory]  │
│                                                                             │
│   Input Vector x            READ_IN / ALLOC             c[0..1]: Input Regs │
│        │                          │                             │           │
│        ▼                          ▼                             ▼           │
│   ┌─────────────┐            ┌──────────────┐          ┌─────────────────┐  │
│   │ x_i * W_ij  │   ───►     │ MULC / ADD   │  ───►    │ Dual-Rail Loops │  │
│   │ Dot Product │            │ Accumulation │          │ [->+<] & Mul-Add│  │
│   └─────────────┘            └──────────────┘          └─────────────────┘  │
│        │                          │                             │           │
│        ▼                          ▼                             ▼           │
│   ┌─────────────┐            ┌──────────────┐          ┌─────────────────┐  │
│   │ acc // {scale:<4} │   ───►     │ DIVC acc, {scale:<3}│  ───►    │ Modulo-Subtract │  │
│   │ Scale Normal│            │ Add bias b_j │          │ Division Tape   │  │
│   └─────────────┘            └──────────────┘          └─────────────────┘  │
│        │                          │                             │           │
│        ▼                          ▼                             ▼           │
│   ┌─────────────┐            ┌──────────────┐          ┌─────────────────┐  │
│   │ Activation  │   ───►     │ {activation.upper():<12} │  ───►    │ Branch/Clamp BF │  │
│   │ {activation.lower():<11} │            │ Target Temp  │          │ [-]+[-] Zeroing │  │
│   └─────────────┘            └──────────────┘          └─────────────────┘  │
│        │                          │                             │           │
│        ▼                          ▼                             ▼           │
│   Next Layer / Output        PRINT_OUT / MOV           c[Output] Register   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
        return diagram
