"""Tests for fixed-point quantization and representation."""

import unittest
import numpy as np
from lobotoml.nn.quantize import (
    QuantizationConfig,
    QuantizedLayer,
    QuantizedModel,
    float_to_fixed,
    fixed_to_float,
)
from lobotoml.nn.model import Sequential
from lobotoml.nn.layers import Dense


class TestQuantization(unittest.TestCase):
    """Test fixed point conversion and layer forward passes."""

    def test_float_to_fixed_conversion(self):
        self.assertEqual(float_to_fixed(0.0, scale=16), 0)
        self.assertEqual(float_to_fixed(1.0, scale=16), 16)
        self.assertEqual(float_to_fixed(0.5, scale=16), 8)
        self.assertEqual(float_to_fixed(-1.25, scale=16), -20)

        arr = np.array([-1.0, 0.0, 0.5, 1.0])
        fixed_arr = float_to_fixed(arr, scale=16)
        np.testing.assert_array_equal(fixed_arr, [-16, 0, 8, 16])

    def test_fixed_to_float_conversion(self):
        self.assertAlmostEqual(fixed_to_float(16, scale=16), 1.0)
        self.assertAlmostEqual(fixed_to_float(8, scale=16), 0.5)
        self.assertAlmostEqual(fixed_to_float(-20, scale=16), -1.25)

    def test_quantized_layer_forward(self):
        # 2 inputs, 1 output: y = relu(2 * x0 - 3 * x1 + 4)
        # in fixed point scale=16:
        # weights = [[32], [-48]], bias = [64]
        layer = QuantizedLayer(
            in_features=2,
            out_features=1,
            weights=[[32], [-48]],
            biases=[64],
            activation="relu",
        )
        # Input: x0=1.0 (16), x1=0.0 (0)
        # acc = 16*32 + 0*(-48) = 512
        # rescaled = 512 // 16 = 32
        # pre_act = 32 + 64 = 96 (Float: 6.0)
        # relu(96) = 96
        out = layer.forward_fixed([16, 0], scale=16)
        self.assertEqual(out, [96])

        # Input: x0=0.0 (0), x1=2.0 (32)
        # acc = 0*32 + 32*(-48) = -1536
        # rescaled = -1536 // 16 = -96
        # pre_act = -96 + 64 = -32 (Float: -2.0)
        # relu(-32) = 0
        out_neg = layer.forward_fixed([0, 32], scale=16)
        self.assertEqual(out_neg, [0])

    def test_quantized_model_from_continuous(self):
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

        self.assertEqual(qmodel.layers[0].weights, [[16, 16], [16, 16]])
        self.assertEqual(qmodel.layers[0].biases, [0, -16])
        self.assertEqual(qmodel.layers[1].weights, [[16], [-32]])
        self.assertEqual(qmodel.layers[1].biases, [0])

        # Test forward pass on XOR input [16, 0]
        out = qmodel.forward_fixed([16, 0])
        self.assertEqual(out, [12])  # 0.75 in scale 16


if __name__ == "__main__":
    unittest.main()
