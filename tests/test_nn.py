"""Tests for neural network primitives, layers, and loss functions."""

import unittest
import numpy as np
from neurofuck.nn.layers import Dense
from neurofuck.nn.activations import ReLU, Sigmoid, HardSigmoid, Linear, Step, get_activation
from neurofuck.nn.loss import MSELoss, BinaryCrossEntropyLoss
from neurofuck.nn.model import Sequential


class TestActivations(unittest.TestCase):
    """Test continuous and fixed-point activation functions."""

    def test_relu(self):
        relu = ReLU()
        x = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
        out = relu.forward(x)
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0, 1.5, 3.0])

        grad = relu.backward(x, np.ones_like(x))
        np.testing.assert_allclose(grad, [0.0, 0.0, 0.0, 1.0, 1.0])

        self.assertEqual(relu.forward_fixed(-10, 16), 0)
        self.assertEqual(relu.forward_fixed(0, 16), 0)
        self.assertEqual(relu.forward_fixed(25, 16), 25)

    def test_linear(self):
        linear = Linear()
        x = np.array([-1.0, 0.0, 2.5])
        np.testing.assert_allclose(linear.forward(x), x)
        self.assertEqual(linear.forward_fixed(-7, 16), -7)

    def test_sigmoid(self):
        sig = Sigmoid()
        x = np.array([0.0])
        np.testing.assert_allclose(sig.forward(x), [0.5])
        self.assertEqual(sig.forward_fixed(0, 16), 8)

    def test_hard_sigmoid(self):
        hs = HardSigmoid()
        x = np.array([-3.0, 0.0, 3.0])
        out = hs.forward(x)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])

        self.assertEqual(hs.forward_fixed(-40, 16), 0)
        self.assertEqual(hs.forward_fixed(0, 16), 8)
        self.assertEqual(hs.forward_fixed(40, 16), 16)

    def test_get_activation(self):
        self.assertIsInstance(get_activation("relu"), ReLU)
        self.assertIsInstance(get_activation("sigmoid"), Sigmoid)
        self.assertIsInstance(get_activation("linear"), Linear)
        self.assertIsInstance(get_activation("hard_sigmoid"), HardSigmoid)


class TestDenseLayer(unittest.TestCase):
    """Test Dense layer forward and backward passes."""

    def test_forward_dimensions(self):
        dense = Dense(3, 2, activation="relu", seed=42)
        x_1d = np.array([1.0, 2.0, 3.0])
        out_1d = dense.forward(x_1d)
        self.assertEqual(out_1d.shape, (2,))

        x_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out_2d = dense.forward(x_2d)
        self.assertEqual(out_2d.shape, (2, 2))

    def test_numerical_gradient(self):
        dense = Dense(2, 2, activation="linear", seed=42)
        x = np.array([[1.5, -0.5]])
        eps = 1e-6

        # Forward and backward
        y = dense.forward(x)
        grad_out = np.array([[1.0, -1.0]])
        dense.backward(grad_out)

        # Check weight gradient numerically
        for i in range(2):
            for j in range(2):
                orig = dense.weights[i, j]
                dense.weights[i, j] = orig + eps
                y_pos = dense.forward(x)
                dense.weights[i, j] = orig - eps
                y_neg = dense.forward(x)
                dense.weights[i, j] = orig

                loss_pos = np.sum(y_pos * grad_out)
                loss_neg = np.sum(y_neg * grad_out)
                num_grad = (loss_pos - loss_neg) / (2 * eps)
                self.assertAlmostEqual(dense.grad_weights[i, j], num_grad, places=4)


class TestTraining(unittest.TestCase):
    """Test model training and convergence on XOR."""

    def test_xor_convergence(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([[0], [1], [1], [0]], dtype=np.float64)

        np.random.seed(0)
        model = Sequential([
            Dense(2, 4, activation="relu", seed=0),
            Dense(4, 1, activation="sigmoid", seed=100),
        ])

        loss_history = model.fit(X, y, epochs=5000, lr=0.1, loss="mse", optimizer="adam")
        self.assertLess(loss_history[-1], 0.01)

        preds = model.predict(X).ravel()
        binary_preds = (preds >= 0.5).astype(int)
        np.testing.assert_array_equal(binary_preds, y.ravel())


if __name__ == "__main__":
    unittest.main()
