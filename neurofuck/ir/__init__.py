"""Intermediate representation for compiling neural operations to Brainfuck."""

from .instructions import (
    Instruction,
    Alloc,
    Free,
    SetConst,
    Move,
    Copy,
    Add,
    Sub,
    AddConst,
    SubConst,
    Mul,
    DivConst,
    ReLU,
    Sigmoid,
    Clamp,
    ReadInput,
    PrintOutput,
    Comment,
)
from .graph import IRProgram
from .visualizer import LayerVisualizer, format_tape_layout

__all__ = [
    "Instruction",
    "Alloc",
    "Free",
    "SetConst",
    "Move",
    "Copy",
    "Add",
    "Sub",
    "AddConst",
    "SubConst",
    "Mul",
    "DivConst",
    "ReLU",
    "Sigmoid",
    "Clamp",
    "ReadInput",
    "PrintOutput",
    "Comment",
    "IRProgram",
    "LayerVisualizer",
    "format_tape_layout",
]
