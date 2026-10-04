"""Tests for Brainfuck arithmetic emitters and macros."""

import unittest
from neurofuck.compiler.memory import TapeMemoryManager
from neurofuck.compiler.bf_emitter import BrainfuckEmitter
from neurofuck.vm.fast_vm import FastBrainfuckVM


class TestEmitter(unittest.TestCase):
    """Test individual Brainfuck arithmetic routines."""

    def setUp(self):
        self.mem = TapeMemoryManager()
        self.em = BrainfuckEmitter(self.mem)

    def _execute(self, out_vars) -> list[int]:
        for v in out_vars:
            self.em.move_to(v)
            self.em.emit_raw(".")
        code = self.em.get_code()
        vm = FastBrainfuckVM(code)
        return vm.run().outputs

    def test_add_and_copy(self):
        self.mem.allocate("a")
        self.mem.allocate("b")
        self.mem.allocate("c")
        self.mem.allocate("t0")

        self.em.set_const("a", 10)
        self.em.copy_cell("a", "b", "t0")
        self.em.add_cell("a", "c", "t0")

        outputs = self._execute(["a", "b", "c"])
        self.assertEqual(outputs, [10, 10, 10])

    def test_multiply_add_const(self):
        self.mem.allocate("x")
        self.mem.allocate("y")
        self.mem.allocate("t0")

        self.em.set_const("x", 7)
        self.em.multiply_add_const("x", "y", 6, "t0")

        outputs = self._execute(["x", "y"])
        self.assertEqual(outputs, [7, 42])

    def test_divide_const(self):
        self.mem.allocate("src")
        self.mem.allocate("dst")
        self.mem.allocate("t0")
        self.mem.allocate("t1")
        self.mem.allocate("t2")

        self.em.set_const("src", 75)
        self.em.divide_const("src", "dst", 16, "t0", "t1", "t2")

        outputs = self._execute(["dst"])
        self.assertEqual(outputs, [75 // 16])

    def test_normalize_dual_rail(self):
        test_pairs = [(12, 5, 7, 0), (5, 12, 0, 7), (8, 8, 0, 0), (0, 0, 0, 0)]
        for p_in, n_in, exp_p, exp_n in test_pairs:
            mem = TapeMemoryManager()
            em = BrainfuckEmitter(mem)
            mem.allocate("p")
            mem.allocate("n")
            mem.allocate("t0")
            mem.allocate("t1")
            mem.allocate("t2")
            mem.allocate("t3")

            em.set_const("p", p_in)
            em.set_const("n", n_in)
            em.normalize_dual_rail("p", "n", "t0", "t1", "t2", "t3")
            em.move_to("p")
            em.emit_raw(".")
            em.move_to("n")
            em.emit_raw(".")

            vm = FastBrainfuckVM(em.get_code())
            res = vm.run()
            self.assertEqual(res.outputs, [exp_p, exp_n], f"Failed for ({p_in}, {n_in})")

    def test_relu_emitter(self):
        for val, exp in [(15, 15), (-10, 0), (0, 0)]:
            mem = TapeMemoryManager()
            em = BrainfuckEmitter(mem)
            mem.allocate("p")
            mem.allocate("n")
            mem.allocate("t0")
            mem.allocate("t1")
            mem.allocate("t2")
            mem.allocate("t3")

            if val >= 0:
                em.set_const("p", val)
                em.zero("n")
            else:
                em.zero("p")
                em.set_const("n", -val)

            em.relu("p", "n", "t0", "t1", "t2", "t3")
            em.move_to("p")
            em.emit_raw(".")
            em.move_to("n")
            em.emit_raw(".")

            vm = FastBrainfuckVM(em.get_code())
            res = vm.run()
            self.assertEqual(res.outputs, [exp, 0])


if __name__ == "__main__":
    unittest.main()
