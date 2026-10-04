"""Brainfuck Virtual Machines for neurofuck execution and verification."""

from .vm import BrainfuckVM, VMConfig, VMResult
from .fast_vm import FastBrainfuckVM

__all__ = [
    "BrainfuckVM",
    "FastBrainfuckVM",
    "VMConfig",
    "VMResult",
]
