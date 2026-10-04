"""Peephole optimizer for Brainfuck code."""

import re


class BrainfuckOptimizer:
    """Optimizes raw Brainfuck sequences using peephole rewrites."""

    @staticmethod
    def optimize(code: str) -> str:
        """Run iterative peephole optimizations on Brainfuck source."""
        # Strip all characters except Brainfuck commands
        clean_code = "".join(c for c in code if c in "><+-.,[]")
        prev_len = len(clean_code) + 1

        while len(clean_code) < prev_len:
            prev_len = len(clean_code)

            # Rule 1: Cancel out opposing pointer movements: >< and <>
            clean_code = clean_code.replace("><", "")
            clean_code = clean_code.replace("<>", "")

            # Rule 2: Cancel out opposing increments and decrements: +- and -+
            clean_code = clean_code.replace("+-", "")
            clean_code = clean_code.replace("-+", "")

            # Rule 3: Redundant clear loops: [-][-] -> [-]
            clean_code = clean_code.replace("[-][-]", "[-]")
            clean_code = clean_code.replace("[+][+]", "[-]")
            clean_code = clean_code.replace("[-][+]", "[-]")

            # Rule 4: Clear immediately after clear/set before loop: e.g. [->+<][-]
            # Note: Do not alter loops containing pointer shifts

        return clean_code
