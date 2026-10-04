"""Tests for model serialization and JSON export/import."""

import unittest
import tempfile
from pathlib import Path
import numpy as np
from lobotoml.nn.model import Sequential
from lobotoml.nn.layers import Dense
from lobotoml.nn.quantize import QuantizedModel, QuantizationConfig


class TestSerialization(unittest.TestCase):
    """Test model persistence to/from JSON."""

    def test_continuous_model_roundtrip(self):
        model = Sequential([
            Dense(2, 3, activation="relu", seed=1),
            Dense(3, 1, activation="sigmoid", seed=2),
        ])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            model.save(temp_path)
            loaded = Sequential.load(temp_path)

            self.assertEqual(len(loaded.layers), 2)
            self.assertEqual(loaded.layers[0].in_features, 2)
            self.assertEqual(loaded.layers[0].out_features, 3)
            self.assertEqual(loaded.layers[0].activation.name, "relu")

            np.testing.assert_allclose(model.layers[0].weights, loaded.layers[0].weights)
            np.testing.assert_allclose(model.layers[0].biases, loaded.layers[0].biases)

            x = np.array([[0.5, 0.2]])
            np.testing.assert_allclose(model.forward(x), loaded.forward(x))
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_quantized_model_dict_roundtrip(self):
        model = Sequential([
            Dense(2, 2, activation="relu"),
            Dense(2, 1, activation="hard_sigmoid"),
        ])
        qconfig = QuantizationConfig(scale=16)
        qmodel = QuantizedModel.from_continuous_model(model, qconfig)

        d = qmodel.to_dict()
        reconstructed = QuantizedModel.from_dict(d)

        self.assertEqual(reconstructed.scale, 16)
        self.assertEqual(len(reconstructed.layers), 2)
        self.assertEqual(qmodel.forward_fixed([16, 16]), reconstructed.forward_fixed([16, 16]))


if __name__ == "__main__":
    unittest.main()
