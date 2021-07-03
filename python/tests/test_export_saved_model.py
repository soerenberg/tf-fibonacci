"""Tests for export_saved_model.py"""
import numpy as np
import pytest
import tensorflow as tf

import export_saved_model


@pytest.fixture(name="loaded_saved_model")
def loaded_saved_model_fixture(tmpdir):
    """Export SavedModel to disk, then load it to memory."""
    export_saved_model.export_saved_model(tmpdir)

    return tf.saved_model.load(str(tmpdir))


@pytest.mark.functional
@pytest.mark.parametrize("sequence,expected",
                         [([0], [0]), ([1], [1]), ([10], [55]),
                          ([34, 5, 25], [5702887, 5, 75025])])
def test_export_saved_model(loaded_saved_model, sequence, expected):
    """Simple functional test that the SavedModel returns correct results."""
    result = loaded_saved_model.signatures["serving_default"](
        tensor=tf.constant(sequence, dtype=tf.int64))

    np.testing.assert_array_equal(result["result"].numpy(), expected)
