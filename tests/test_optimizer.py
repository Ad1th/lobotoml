"""Tests for Brainfuck peephole optimizer."""

import unittest
from neurofuck.compiler.optimizer import BrainfuckOptimizer
from neurofuck.vm.fast_vm import FastBrainfuckVM


class TestOptimizer(unittest.TestCase):
    """Test peephole optimizer patterns and semantic preservation."""

    def test_cancellations(self):
        code = "+++---"
        self.assertEqual(BrainfuckOptimizer.optimize(code), "")

        code = ">>>><<<<"
        self.assertEqual(BrainfuckOptimizer.optimize(code), "")

        code = "[-][-]"
        self.assertEqual(BrainfuckOptimizer.optimize(code), "[-]")

    def test_semantic_preservation(self):
        # Program computes (7 * 6) + 3 = 45
        unopt = "+++++++[>++++++<-]>+++><><+-."
        opt = BrainfuckOptimizer.optimize(unopt)

        self.assertLess(len(opt), len(unopt))

        vm_unopt = FastBrainfuckVM(unopt)
        vm_opt = FastBrainfuckVM(opt)

        self.assertEqual(vm_unopt.run().outputs, [45])
        self.assertEqual(vm_opt.run().outputs, [45])


if __name__ == "__main__":
    unittest.main()
