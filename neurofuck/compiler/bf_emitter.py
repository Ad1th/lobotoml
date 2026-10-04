"""Brainfuck code emitter implementing fixed-point neural operations."""

from typing import List, Optional
from .memory import TapeMemoryManager


def rel_move(from_cell: int, to_cell: int) -> str:
    """Generate Brainfuck pointer movements from one absolute cell to another."""
    delta = to_cell - from_cell
    if delta > 0:
        return ">" * delta
    elif delta < 0:
        return "<" * (-delta)
    return ""


class BrainfuckEmitter:
    """Emits optimized Brainfuck code sequences for arithmetic and neural network operations."""

    def __init__(self, memory: TapeMemoryManager):
        self.mem = memory
        self._code_chunks: List[str] = []

    def get_code(self) -> str:
        """Get emitted Brainfuck code."""
        return "".join(self._code_chunks)

    def emit_raw(self, bf_code: str) -> None:
        """Append raw Brainfuck string."""
        if bf_code:
            self._code_chunks.append(bf_code)

    def move_to(self, var: str) -> None:
        """Navigate data pointer to variable cell."""
        code = self.mem.move_to(var)
        self.emit_raw(code)

    def zero(self, var: str) -> None:
        """Clear variable cell to 0: [-]."""
        self.move_to(var)
        self.emit_raw("[-]")

    def set_const(self, var: str, val: int) -> None:
        """Set variable cell to non-negative constant integer."""
        if val < 0:
            raise ValueError(f"Direct set_const requires non-negative value, got {val}")
        self.zero(var)
        if val > 0:
            self.emit_raw("+" * val)

    def add_const(self, var: str, val: int) -> None:
        """Add constant to variable cell."""
        if val == 0:
            return
        self.move_to(var)
        if val > 0:
            self.emit_raw("+" * val)
        else:
            self.emit_raw("-" * (-val))

    def move_cell(self, src: str, dst: str) -> None:
        """Destructively move value from src to dst (dst is zeroed first)."""
        self.zero(dst)
        src_c = self.mem.get_cell(src)
        dst_c = self.mem.get_cell(dst)

        self.move_to(src)
        self.emit_raw(f"[{rel_move(src_c, dst_c)}+{rel_move(dst_c, src_c)}-]")
        self.mem._current_ptr = src_c

    def add_cell(self, src: str, dst: str, temp: str) -> None:
        """Non-destructively add src to dst: dst += src (uses temp cell)."""
        self.zero(temp)
        src_c = self.mem.get_cell(src)
        dst_c = self.mem.get_cell(dst)
        tmp_c = self.mem.get_cell(temp)

        # src[dst+ temp+ src-]
        self.move_to(src)
        self.emit_raw(f"[{rel_move(src_c, dst_c)}+{rel_move(dst_c, tmp_c)}+{rel_move(tmp_c, src_c)}-]")
        self.mem._current_ptr = src_c

        # temp[src+ temp-]
        self.move_to(temp)
        self.emit_raw(f"[{rel_move(tmp_c, src_c)}+{rel_move(src_c, tmp_c)}-]")
        self.mem._current_ptr = tmp_c

    def copy_cell(self, src: str, dst: str, temp: str) -> None:
        """Non-destructively copy src to dst: dst = src."""
        self.zero(dst)
        self.add_cell(src, dst, temp)

    def multiply_add_const(self, src: str, dst: str, factor: int, temp: str) -> None:
        """Non-destructively compute: dst += src * factor."""
        if factor == 0:
            return
        if factor < 0:
            raise ValueError("multiply_add_const factor must be non-negative in dual-rail")

        self.zero(temp)
        src_c = self.mem.get_cell(src)
        dst_c = self.mem.get_cell(dst)
        tmp_c = self.mem.get_cell(temp)

        plus_str = "+" * factor

        # src[dst +*factor temp+ src-]
        self.move_to(src)
        self.emit_raw(f"[{rel_move(src_c, dst_c)}{plus_str}{rel_move(dst_c, tmp_c)}+{rel_move(tmp_c, src_c)}-]")
        self.mem._current_ptr = src_c

        # temp[src+ temp-]
        self.move_to(temp)
        self.emit_raw(f"[{rel_move(tmp_c, src_c)}+{rel_move(src_c, tmp_c)}-]")
        self.mem._current_ptr = tmp_c

    def divide_const(self, src: str, dst_quotient: str, divisor: int, rem_var: str, flag_var: str, backup_var: str) -> None:
        """Compute integer quotient: dst_quotient = src // divisor (src is consumed)."""
        if divisor <= 0:
            raise ValueError(f"Divisor must be positive integer, got {divisor}")

        self.zero(dst_quotient)
        self.set_const(rem_var, divisor)
        self.zero(flag_var)
        self.zero(backup_var)

        src_c = self.mem.get_cell(src)
        q_c = self.mem.get_cell(dst_quotient)
        rem_c = self.mem.get_cell(rem_var)
        flag_c = self.mem.get_cell(flag_var)
        backup_c = self.mem.get_cell(backup_var)

        div_plus = "+" * divisor

        self.move_to(src)
        loop = (
            f"["
            f"-{rel_move(src_c, rem_c)}-"
            f"{rel_move(rem_c, flag_c)}[-]+"
            f"{rel_move(flag_c, rem_c)}["
            f"{rel_move(rem_c, flag_c)}[-]"
            f"{rel_move(flag_c, backup_c)}+"
            f"{rel_move(backup_c, rem_c)}-"
            f"]"
            f"{rel_move(rem_c, backup_c)}["
            f"{rel_move(backup_c, rem_c)}+"
            f"{rel_move(rem_c, backup_c)}-"
            f"]"
            f"{rel_move(backup_c, flag_c)}["
            f"{rel_move(flag_c, rem_c)}{div_plus}"
            f"{rel_move(rem_c, q_c)}+"
            f"{rel_move(q_c, flag_c)}-"
            f"]"
            f"{rel_move(flag_c, src_c)}"
            f"]"
        )
        self.emit_raw(loop)
        self.mem._current_ptr = src_c

        self.zero(rem_var)
        self.zero(flag_var)
        self.zero(backup_var)

    def normalize_dual_rail(
        self,
        pos_var: str,
        neg_var: str,
        flag: str,
        inv: str,
        saved_neg: str,
        temp: str,
    ) -> None:
        """Normalize dual-rail pair (pos_var, neg_var) so at most one is non-zero."""
        self.zero(saved_neg)
        self.zero(flag)
        self.zero(inv)
        self.zero(temp)

        pos_c = self.mem.get_cell(pos_var)
        neg_c = self.mem.get_cell(neg_var)
        flag_c = self.mem.get_cell(flag)
        inv_c = self.mem.get_cell(inv)
        saved_c = self.mem.get_cell(saved_neg)
        temp_c = self.mem.get_cell(temp)

        # Start from current position and move to neg_c
        self.move_to(neg_var)
        loop = (
            f"["
            f"-{rel_move(neg_c, flag_c)}[-]{rel_move(flag_c, inv_c)}[-]+"
            f"{rel_move(inv_c, pos_c)}[{rel_move(pos_c, temp_c)}+{rel_move(temp_c, pos_c)}-]"
            f"{rel_move(pos_c, temp_c)}[{rel_move(temp_c, pos_c)}+{rel_move(pos_c, flag_c)}[-]+{rel_move(flag_c, inv_c)}[-]{rel_move(inv_c, temp_c)}-]"
            f"{rel_move(temp_c, flag_c)}[{rel_move(flag_c, pos_c)}-{rel_move(pos_c, flag_c)}-]"
            f"{rel_move(flag_c, inv_c)}[{rel_move(inv_c, neg_c)}+[{rel_move(neg_c, saved_c)}+{rel_move(saved_c, neg_c)}-]{rel_move(neg_c, inv_c)}-]"
            f"{rel_move(inv_c, neg_c)}]"
        )
        self.emit_raw(loop)
        self.mem._current_ptr = neg_c

        # Restore saved_neg -> neg_var
        self.move_to(saved_neg)
        self.emit_raw(f"[{rel_move(saved_c, neg_c)}+{rel_move(neg_c, saved_c)}-]")
        self.mem._current_ptr = saved_c

    def clamp_max(
        self,
        var: str,
        max_val: int,
        t0: str,
        t1: str,
        t2: str,
        t3: str,
        t4: str,
    ) -> None:
        """Clamp non-negative variable: if var > max_val: var = max_val."""
        self.set_const(t0, max_val)
        self.normalize_dual_rail(var, t0, t1, t2, t3, t4)

        var_c = self.mem.get_cell(var)
        t0_c = self.mem.get_cell(t0)

        self.set_const(var, max_val)
        self.move_to(t0)
        self.emit_raw(f"[{rel_move(t0_c, var_c)}-{rel_move(var_c, t0_c)}-]")
        self.mem._current_ptr = t0_c

        self.zero(t1)
        self.zero(t2)
        self.zero(t3)
        self.zero(t4)

    def relu(
        self,
        pos_var: str,
        neg_var: str,
        flag: str,
        inv: str,
        saved_neg: str,
        temp: str,
    ) -> None:
        """ReLU activation on dual-rail: max(0, x)."""
        self.normalize_dual_rail(pos_var, neg_var, flag, inv, saved_neg, temp)
        self.zero(neg_var)

    def hard_sigmoid(
        self,
        pos_var: str,
        neg_var: str,
        out_pos: str,
        scale: int,
        scratch: List[str],
    ) -> None:
        """Hard sigmoid activation on dual-rail: clamp(x // 4 + scale // 2, 0, scale)."""
        s0, s1, s2, s3, s4, s5 = scratch[:6]

        # 1. Normalize input
        self.normalize_dual_rail(pos_var, neg_var, s0, s1, s2, s3)

        # 2. Divide pos_var // 4 -> s0, neg_var // 4 -> s1
        self.divide_const(pos_var, s0, 4, s2, s3, s4)
        self.divide_const(neg_var, s1, 4, s2, s3, s4)

        # 3. Pre-activation Z_pos = s0 + scale // 2, Z_neg = s1
        half_scale = scale // 2
        self.add_const(s0, half_scale)

        # 4. Normalize (s0, s1)
        self.normalize_dual_rail(s0, s1, s2, s3, s4, s5)

        # 5. Lower bound clamp: if s1 > 0, then s0 is 0. zero(s1) leaves s0 = 0
        self.zero(s1)

        # 6. Upper bound clamp: clamp s0 <= scale
        self.clamp_max(s0, scale, s1, s2, s3, s4, s5)

        # 7. Move result to out_pos
        self.move_cell(s0, out_pos)
