"""Instruction definitions for Neurofuck Intermediate Representation (NIR)."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Instruction:
    """Base class for all NIR instructions."""
    comment: str = ""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass
class Alloc(Instruction):
    """Allocate a variable symbol on the tape."""
    name: str = ""
    initial_value: int = 0

    def __str__(self) -> str:
        s = f"ALLOC {self.name} = {self.initial_value}"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class Free(Instruction):
    """Deallocate/free a variable symbol from the tape."""
    name: str = ""

    def __str__(self) -> str:
        s = f"FREE {self.name}"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class SetConst(Instruction):
    """Set variable to a constant integer value: target = value."""
    target: str = ""
    value: int = 0

    def __str__(self) -> str:
        s = f"SET {self.target}, {self.value}"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class AddConst(Instruction):
    """Add constant to variable: target += value."""
    target: str = ""
    value: int = 0

    def __str__(self) -> str:
        s = f"ADDC {self.target}, {self.value}"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class SubConst(Instruction):
    """Subtract constant from variable: target -= value."""
    target: str = ""
    value: int = 0

    def __str__(self) -> str:
        s = f"SUBC {self.target}, {self.value}"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class Move(Instruction):
    """Destructive move: target = source; source = 0."""
    target: str = ""
    source: str = ""

    def __str__(self) -> str:
        s = f"MOV {self.target}, {self.source}"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class Copy(Instruction):
    """Non-destructive copy: target = source."""
    target: str = ""
    source: str = ""

    def __str__(self) -> str:
        s = f"CPY {self.target}, {self.source}"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class Add(Instruction):
    """Non-destructive add: target += source."""
    target: str = ""
    source: str = ""

    def __str__(self) -> str:
        s = f"ADD {self.target}, {self.source}"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class Sub(Instruction):
    """Non-destructive sub: target -= source."""
    target: str = ""
    source: str = ""

    def __str__(self) -> str:
        s = f"SUB {self.target}, {self.source}"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class Mul(Instruction):
    """Multiplication: target = src1 * src2."""
    target: str = ""
    src1: str = ""
    src2: str = ""

    def __str__(self) -> str:
        s = f"MUL {self.target}, {self.src1}, {self.src2}"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class MulConst(Instruction):
    """Multiplication by constant: target = source * const."""
    target: str = ""
    source: str = ""
    const: int = 0

    def __str__(self) -> str:
        s = f"MULC {self.target}, {self.source}, {self.const}"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class DivConst(Instruction):
    """Integer division by constant: target = source // divisor."""
    target: str = ""
    source: str = ""
    divisor: int = 1

    def __str__(self) -> str:
        s = f"DIVC {self.target}, {self.source}, {self.divisor}"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class ReLU(Instruction):
    """ReLU activation: target = max(0, source)."""
    target: str = ""
    source: str = ""

    def __str__(self) -> str:
        s = f"RELU {self.target}, {self.source}"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class Sigmoid(Instruction):
    """Hard-sigmoid activation in fixed-point: target = clamp(source // 4 + scale // 2, 0, scale)."""
    target: str = ""
    source: str = ""
    scale: int = 64

    def __str__(self) -> str:
        s = f"SIGMOID {self.target}, {self.source} (scale={self.scale})"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class Clamp(Instruction):
    """Clamp integer value between min_val and max_val: target = clamp(source, min_val, max_val)."""
    target: str = ""
    source: str = ""
    min_val: int = 0
    max_val: int = 64

    def __str__(self) -> str:
        s = f"CLAMP {self.target}, {self.source}, [{self.min_val}, {self.max_val}]"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class ReadInput(Instruction):
    """Read a scaled integer input from the environment: target = input[index]."""
    target: str = ""
    index: int = 0

    def __str__(self) -> str:
        s = f"READ_IN {self.target}, in[{self.index}]"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class PrintOutput(Instruction):
    """Output the integer value of source as prediction[index]."""
    source: str = ""
    index: int = 0

    def __str__(self) -> str:
        s = f"PRINT_OUT {self.source}, out[{self.index}]"
        return f"{s:<30} # {self.comment}" if self.comment else s


@dataclass
class Comment(Instruction):
    """Non-executable documentary comment."""
    text: str = ""

    def __str__(self) -> str:
        return f"# {self.text}"
