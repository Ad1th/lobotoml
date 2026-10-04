"""Tests for Brainfuck Virtual Machine execution and safety features."""

import unittest
from neurofuck.vm.vm import BrainfuckVM, VMConfig
from neurofuck.vm.fast_vm import FastBrainfuckVM


class TestBrainfuckVM(unittest.TestCase):
    """Test standard and fast Brainfuck VM engines."""

    def test_hello_world(self):
        # Classic Brainfuck Hello World
        hello_code = (
            "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>"
            "---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++."
        )
        vm = BrainfuckVM(hello_code)
        res = vm.run()
        self.assertEqual(res.output_str, "Hello World!\n")

        fast_vm = FastBrainfuckVM(hello_code)
        fast_res = fast_vm.run()
        self.assertEqual(fast_res.output_str, "Hello World!\n")

    def test_integer_io(self):
        # Read two integers from input, add them, and output sum
        # code: ,>,<[>+<-]>.
        add_code = ",>,<[>+<-]>."
        vm = BrainfuckVM(add_code)
        res = vm.run(inputs=[15, 27])
        self.assertEqual(res.outputs, [42])

        fast_vm = FastBrainfuckVM(add_code)
        fast_res = fast_vm.run(inputs=[15, 27])
        self.assertEqual(fast_res.outputs, [42])

    def test_infinite_loop_prevention(self):
        infinite_code = "+[]"
        vm = BrainfuckVM(infinite_code, config=VMConfig(max_steps=1000))
        with self.assertRaises(TimeoutError):
            vm.run()

        fast_vm = FastBrainfuckVM(infinite_code, config=VMConfig(max_steps=1000))
        with self.assertRaises(TimeoutError):
            fast_vm.run()

    def test_pointer_underflow_detection(self):
        underflow_code = "<+"
        vm = BrainfuckVM(underflow_code)
        with self.assertRaises(IndexError):
            vm.run()


if __name__ == "__main__":
    unittest.main()
