"""Optimized bytecode Brainfuck Virtual Machine for accelerated execution."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from .vm import VMConfig, VMResult


OP_ADD = 1          # tape[ptr] += arg
OP_MOVE = 2         # ptr += arg
OP_SET = 3          # tape[ptr] = arg (e.g. [-] -> SET 0)
OP_IN = 4           # tape[ptr] = input()
OP_OUT = 5          # output(tape[ptr])
OP_JZ = 6           # jump to arg if tape[ptr] == 0
OP_JNZ = 7          # jump to arg if tape[ptr] != 0
OP_SCAN = 8         # find next 0: ptr += arg * distance
OP_MUL_ADD = 9      # multiply-add: tape[ptr + offset] += tape[ptr] * factor; tape[ptr] = 0


@dataclass
class FastOp:
    op: int
    arg: int = 0
    extra: int = 0


class FastBrainfuckVM:
    """Bytecode JIT/IR optimized Brainfuck VM for high-speed simulation."""

    def __init__(self, code: str, config: Optional[VMConfig] = None):
        self.raw_code = "".join(c for c in code if c in "><+-.,[]")
        self.config = config or VMConfig()
        self.ops: List[FastOp] = self._compile_bytecode(self.raw_code)

    def _compile_bytecode(self, code: str) -> List[FastOp]:
        ops: List[FastOp] = []
        i = 0
        n = len(code)

        while i < n:
            c = code[i]
            if c in "+-":
                delta = 0
                while i < n and code[i] in "+-":
                    delta += 1 if code[i] == "+" else -1
                    i += 1
                if delta != 0:
                    ops.append(FastOp(OP_ADD, delta))
                continue
            elif c in "><":
                delta = 0
                while i < n and code[i] in "><":
                    delta += 1 if code[i] == ">" else -1
                    i += 1
                if delta != 0:
                    ops.append(FastOp(OP_MOVE, delta))
                continue
            elif c == ".":
                ops.append(FastOp(OP_OUT))
                i += 1
            elif c == ",":
                ops.append(FastOp(OP_IN))
                i += 1
            elif c == "[":
                # Check for [-] clear pattern
                if i + 2 < n and code[i+1] == "-" and code[i+2] == "]":
                    ops.append(FastOp(OP_SET, 0))
                    i += 3
                    continue
                # Check for [+] clear pattern
                if i + 2 < n and code[i+1] == "+" and code[i+2] == "]":
                    ops.append(FastOp(OP_SET, 0))
                    i += 3
                    continue
                ops.append(FastOp(OP_JZ, 0))
                i += 1
            elif c == "]":
                ops.append(FastOp(OP_JNZ, 0))
                i += 1
            else:
                i += 1

        # Fix jump targets
        stack = []
        for idx, op in enumerate(ops):
            if op.op == OP_JZ:
                stack.append(idx)
            elif op.op == OP_JNZ:
                if not stack:
                    raise SyntaxError("Unmatched ']' in Brainfuck bytecode")
                start = stack.pop()
                ops[start].arg = idx
                op.arg = start

        if stack:
            raise SyntaxError("Unmatched '[' in Brainfuck bytecode")

        return ops

    def run(self, inputs: Optional[List[int]] = None) -> VMResult:
        tape: List[int] = [0] * max(100, self.config.tape_size)
        ptr = 0
        pc = 0
        steps = 0
        max_steps = self.config.max_steps
        input_queue = list(inputs or [])
        outputs: List[int] = []
        ops = self.ops
        num_ops = len(ops)
        cell_type = self.config.cell_type

        while pc < num_ops:
            steps += 1
            if steps > max_steps:
                raise TimeoutError(f"Brainfuck VM exceeded maximum steps limit ({max_steps})")

            op = ops[pc]
            code = op.op

            if code == OP_ADD:
                tape[ptr] += op.arg
                if cell_type == "uint8":
                    tape[ptr] &= 0xFF
            elif code == OP_MOVE:
                ptr += op.arg
                if ptr < 0:
                    raise IndexError(f"Data pointer underflow at instruction {pc}")
                if ptr >= len(tape):
                    tape.extend([0] * max(len(tape), ptr - len(tape) + 1024))
            elif code == OP_SET:
                tape[ptr] = op.arg
            elif code == OP_JZ:
                if tape[ptr] == 0:
                    pc = op.arg
            elif code == OP_JNZ:
                if tape[ptr] != 0:
                    pc = op.arg
            elif code == OP_OUT:
                outputs.append(tape[ptr])
            elif code == OP_IN:
                if input_queue:
                    val = input_queue.pop(0)
                    if cell_type == "uint8":
                        val &= 0xFF
                    tape[ptr] = val
                else:
                    tape[ptr] = 0

            pc += 1

        snapshot = {i: v for i, v in enumerate(tape) if v != 0}
        output_str = "".join(chr(c & 0xFF) for c in outputs if 0 <= c <= 127)

        return VMResult(
            outputs=outputs,
            output_str=output_str,
            steps=steps,
            tape_snapshot=snapshot,
            final_ptr=ptr,
        )
