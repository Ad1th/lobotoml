"""Tests for bit-exact prediction equivalence between Python and Brainfuck."""

import unittest
import numpy as np
from lobotoml.nn.model import Sequential
from lobotoml.nn.layers import Dense
from lobotoml.nn.quantize import QuantizedModel, QuantizationConfig, float_to_fixed
from lobotoml.compiler.compiler import LobotoMLCompiler
from lobotoml.vm.fast_vm import FastBrainfuckVM


class TestEquivalence(unittest.TestCase):
    """Verify bit-exact predictions between Python Quantized inference and Brainfuck VM."""

    def test_xor_equivalence(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([[0], [1], [1], [0]], dtype=np.float64)

        np.random.seed(0)
        model = Sequential([
            Dense(2, 4, activation="relu", seed=0),
            Dense(4, 1, activation="sigmoid", seed=100),
        ])
        model.fit(X, y, epochs=5000, lr=0.1, loss="mse", optimizer="adam")

        qconfig = QuantizationConfig(scale=16)
        qmodel = QuantizedModel.from_continuous_model(model, qconfig)

        compiler = LobotoMLCompiler(qconfig)
        bf_code = compiler.compile(qmodel, optimize=True)
        vm = FastBrainfuckVM(bf_code)

        for x, target in zip(X, y):
            fixed_in = [float_to_fixed(v, 16) for v in x]
            py_fixed_out = qmodel.forward_fixed(fixed_in)
            res = vm.run(inputs=fixed_in)

            self.assertEqual(
                py_fixed_out,
                res.outputs,
                f"Mismatch for input {x}: Py {py_fixed_out} != BF {res.outputs}"
            )
            # Check classification correctness
            pred_class = int(res.outputs[0] >= 8)
            self.assertEqual(pred_class, int(target[0]))

    def test_compiler_determinism(self):
        """Compiling the exact same model twice must yield identical Brainfuck strings."""
        model = Sequential([
            Dense(2, 2, activation="relu"),
            Dense(2, 1, activation="hard_sigmoid"),
        ])
        qconfig = QuantizationConfig(scale=16)
        qmodel = QuantizedModel.from_continuous_model(model, qconfig)

        compiler = LobotoMLCompiler(qconfig)
        code1 = compiler.compile(qmodel, optimize=True)
        code2 = compiler.compile(qmodel, optimize=True)

        self.assertEqual(code1, code2)


if __name__ == "__main__":
    unittest.main()
