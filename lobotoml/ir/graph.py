"""Intermediate representation container and reference interpreter."""

from typing import List, Dict, Any, Optional
from .instructions import (
    Instruction,
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
    Clamp,
    ReadInput,
    PrintOutput,
    Comment,
)


class IRProgram:
    """A linear sequence of LobotoML Intermediate Representation (LIR) instructions."""

    def __init__(self, instructions: Optional[List[Instruction]] = None):
        self.instructions: List[Instruction] = instructions or []
        self.metadata: Dict[str, Any] = {}

    def emit(self, instruction: Instruction) -> None:
        """Append an instruction to the program."""
        self.instructions.append(instruction)

    def comment(self, text: str) -> None:
        """Append a comment."""
        self.instructions.append(Comment(text=text))

    def __iter__(self):
        return iter(self.instructions)

    def __len__(self) -> int:
        return len(self.instructions)

    def to_assembly(self) -> str:
        """Format program as readable pseudo-assembly."""
        lines = []
        for inst in self.instructions:
            lines.append(str(inst))
        return "\n".join(lines)

    def execute(self, inputs: List[int]) -> List[int]:
        """Reference interpreter for NIR instructions.
        
        Args:
            inputs: List of integer inputs.
        Returns:
            List of integer outputs produced by PRINT_OUT instructions.
        """
        env: Dict[str, int] = {}
        outputs: List[int] = []

        for inst in self.instructions:
            if isinstance(inst, Comment):
                continue
            elif isinstance(inst, Alloc):
                env[inst.name] = inst.initial_value
            elif isinstance(inst, Free):
                env.pop(inst.name, None)
            elif isinstance(inst, SetConst):
                env[inst.target] = inst.value
            elif isinstance(inst, AddConst):
                env[inst.target] = env.get(inst.target, 0) + inst.value
            elif isinstance(inst, SubConst):
                env[inst.target] = env.get(inst.target, 0) - inst.value
            elif isinstance(inst, Move):
                env[inst.target] = env.get(inst.source, 0)
                env[inst.source] = 0
            elif isinstance(inst, Copy):
                env[inst.target] = env.get(inst.source, 0)
            elif isinstance(inst, Add):
                env[inst.target] = env.get(inst.target, 0) + env.get(inst.source, 0)
            elif isinstance(inst, Sub):
                env[inst.target] = env.get(inst.target, 0) - env.get(inst.source, 0)
            elif isinstance(inst, Mul):
                env[inst.target] = env.get(inst.src1, 0) * env.get(inst.src2, 0)
            elif isinstance(inst, MulConst):
                env[inst.target] = env.get(inst.source, 0) * inst.const
            elif isinstance(inst, DivConst):
                val = env.get(inst.source, 0)
                div = inst.divisor
                if div == 0:
                    raise ZeroDivisionError("Division by zero in NIR DivConst")
                # Integer division truncating towards zero
                if val >= 0:
                    env[inst.target] = val // div
                else:
                    env[inst.target] = -((-val) // div)
            elif isinstance(inst, ReLU):
                env[inst.target] = max(0, env.get(inst.source, 0))
            elif isinstance(inst, Sigmoid):
                val = env.get(inst.source, 0)
                # Hard-sigmoid: clamp(val // 4 + scale // 2, 0, scale)
                res = (val // 4) + (inst.scale // 2)
                env[inst.target] = max(0, min(inst.scale, res))
            elif isinstance(inst, Clamp):
                val = env.get(inst.source, 0)
                env[inst.target] = max(inst.min_val, min(inst.max_val, val))
            elif isinstance(inst, ReadInput):
                if inst.index < len(inputs):
                    env[inst.target] = inputs[inst.index]
                else:
                    env[inst.target] = 0
            elif isinstance(inst, PrintOutput):
                outputs.append(env.get(inst.source, 0))
            else:
                raise NotImplementedError(f"Unknown instruction type: {type(inst)}")

        return outputs
