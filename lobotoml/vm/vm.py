"""Standard Brainfuck Virtual Machine with full instrumentation and debugging."""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any


@dataclass
class VMConfig:
    """Configuration for Brainfuck execution environment.
    
    Attributes:
        tape_size: Initial memory tape size (dynamically expands if needed).
        cell_type: 'signed_int' (default 32-bit/64-bit integer), 'uint8' (8-bit wrap), 'int16'
        max_steps: Maximum instruction steps before timeout (prevents infinite loops).
    """
    tape_size: int = 30000
    cell_type: str = "signed_int"
    max_steps: int = 50_000_000


@dataclass
class VMResult:
    """Result of Brainfuck VM execution.
    
    Attributes:
        outputs: List of integer values output by the program.
        output_str: String representation of decoded character outputs.
        steps: Total instruction steps executed.
        tape_snapshot: Non-zero tape memory cells at termination.
        final_ptr: Final data pointer position.
    """
    outputs: List[int]
    output_str: str
    steps: int
    tape_snapshot: Dict[int, int]
    final_ptr: int


class BrainfuckVM:
    """Reference Brainfuck Virtual Machine in pure Python."""

    def __init__(self, code: str, config: Optional[VMConfig] = None):
        self.code = "".join(c for c in code if c in "><+-.,[]")
        self.config = config or VMConfig()
        self.tape: List[int] = [0] * max(100, self.config.tape_size)
        self.ptr: int = 0
        self.pc: int = 0
        self.step_count: int = 0
        self.jump_map: Dict[int, int] = self._precompute_jumps()

    def _precompute_jumps(self) -> Dict[int, int]:
        """Precompute matching bracket jump targets."""
        stack = []
        jump_map = {}
        for idx, char in enumerate(self.code):
            if char == "[":
                stack.append(idx)
            elif char == "]":
                if not stack:
                    raise SyntaxError(f"Unmatched closing bracket ']' at code index {idx}")
                start = stack.pop()
                jump_map[start] = idx
                jump_map[idx] = start
        if stack:
            raise SyntaxError(f"Unmatched opening bracket '[' at code index {stack[-1]}")
        return jump_map

    def _ensure_tape(self, index: int) -> None:
        """Dynamically expand tape if pointer exceeds bounds."""
        if index >= len(self.tape):
            extension = max(len(self.tape), index - len(self.tape) + 1024)
            self.tape.extend([0] * extension)

    def run(self, inputs: Optional[List[int]] = None) -> VMResult:
        """Execute the Brainfuck program.
        
        Args:
            inputs: List of integer input values to supply to `,` instructions.
        Returns:
            VMResult containing outputs, step count, and tape snapshot.
        """
        input_queue = list(inputs or [])
        outputs: List[int] = []
        code_len = len(self.code)
        max_steps = self.config.max_steps
        cell_type = self.config.cell_type

        while self.pc < code_len:
            self.step_count += 1
            if self.step_count > max_steps:
                raise TimeoutError(f"Brainfuck VM exceeded maximum execution steps limit ({max_steps})")

            cmd = self.code[self.pc]

            if cmd == ">":
                self.ptr += 1
                self._ensure_tape(self.ptr)
            elif cmd == "<":
                if self.ptr > 0:
                    self.ptr -= 1
                else:
                    raise IndexError(f"Data pointer underflow (attempted to move left of cell 0 at pc={self.pc})")
            elif cmd == "+":
                self.tape[self.ptr] += 1
                if cell_type == "uint8":
                    self.tape[self.ptr] &= 0xFF
            elif cmd == "-":
                self.tape[self.ptr] -= 1
                if cell_type == "uint8":
                    self.tape[self.ptr] &= 0xFF
            elif cmd == ".":
                outputs.append(self.tape[self.ptr])
            elif cmd == ",":
                if input_queue:
                    val = input_queue.pop(0)
                    if cell_type == "uint8":
                        val &= 0xFF
                    self.tape[self.ptr] = val
                else:
                    self.tape[self.ptr] = 0
            elif cmd == "[":
                if self.tape[self.ptr] == 0:
                    self.pc = self.jump_map[self.pc]
            elif cmd == "]":
                if self.tape[self.ptr] != 0:
                    self.pc = self.jump_map[self.pc]

            self.pc += 1

        # Build snapshot of active cells
        snapshot = {i: v for i, v in enumerate(self.tape) if v != 0}
        
        # Build output character string where applicable
        output_str = "".join(chr(c & 0xFF) for c in outputs if 0 <= c <= 127)

        return VMResult(
            outputs=outputs,
            output_str=output_str,
            steps=self.step_count,
            tape_snapshot=snapshot,
            final_ptr=self.ptr,
        )
