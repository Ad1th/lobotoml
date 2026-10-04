"""Tests for NIR instructions and reference interpreter."""

import unittest
import numpy as np
from lobotoml.ir.instructions import (
    Alloc, Free, SetConst, AddConst, SubConst, Move, Copy,
    Add, Sub, Mul, MulConst, DivConst, ReLU, Sigmoid, ReadInput, PrintOutput
)
from lobotoml.ir.graph import IRProgram
from lobotoml.compiler.ir_generator import NNToIRCompiler
from lobotoml.nn.model import Sequential
from lobotoml.nn.layers import Dense
from lobotoml.nn.quantize import QuantizedModel, QuantizationConfig


class TestIR(unittest.TestCase):
    """Test NIR instructions and interpreter execution."""

    def test_basic_ir_execution(self):
        prog = IRProgram()
        prog.emit(Alloc("x", 10))
        prog.emit(Alloc("y", 20))
        prog.emit(Add("y", "x"))          # y = 20 + 10 = 30
        prog.emit(DivConst("z", "y", 5))  # z = 30 // 5 = 6
        prog.emit(PrintOutput("z", 0))

        outputs = prog.execute(inputs=[])
        self.assertEqual(outputs, [6])

    def test_relu_sigmoid_ir(self):
        prog = IRProgram()
        prog.emit(Alloc("a", -10))
        prog.emit(ReLU("r1", "a"))       # r1 = max(0, -10) = 0
        prog.emit(Alloc("b", 15))
        prog.emit(ReLU("r2", "b"))       # r2 = max(0, 15) = 15
        prog.emit(Sigmoid("s", "b", scale=16)) # s = clamp(15 // 4 + 8, 0, 16) = clamp(3 + 8, 0, 16) = 11

        prog.emit(PrintOutput("r1", 0))
        prog.emit(PrintOutput("r2", 1))
        prog.emit(PrintOutput("s", 2))

        outputs = prog.execute(inputs=[])
        self.assertEqual(outputs, [0, 15, 11])

    def test_nn_to_ir_compiler_equivalence(self):
        model = Sequential([
            Dense(2, 2, activation="relu"),
            Dense(2, 1, activation="hard_sigmoid"),
        ])
        model.layers[0].weights = np.array([[1.0, 1.0], [1.0, 1.0]])
        model.layers[0].biases = np.array([0.0, -1.0])
        model.layers[1].weights = np.array([[1.0], [-2.0]])
        model.layers[1].biases = np.array([0.0])

        qconfig = QuantizationConfig(scale=16)
        qmodel = QuantizedModel.from_continuous_model(model, qconfig)

        ir_compiler = NNToIRCompiler(qmodel)
        prog = ir_compiler.compile()

        inputs = [16, 0]
        ir_out = prog.execute(inputs)
        q_out = qmodel.forward_fixed(inputs)

        self.assertEqual(ir_out, q_out)


if __name__ == "__main__":
    unittest.main()
