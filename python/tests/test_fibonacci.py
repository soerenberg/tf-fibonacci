"""Tests for fibonacci.py module"""
import numpy as np
import pytest
import tensorflow as tf

import fibonacci


@pytest.mark.parametrize("dtype", [tf.int32, tf.int64])
@pytest.mark.parametrize("sequence,expected",
                         [([0], [0]), ([1], [1]), ([10], [55]),
                          ([34, 5, 25], [5702887, 5, 75025])])
def test_fibonacci_numbers(sequence, expected, dtype):
    """Test for fibonacci_numbers function"""
    result = fibonacci.fibonacci_numbers(tensor=tf.constant(sequence, dtype))

    assert result.dtype == dtype
    np.testing.assert_array_equal(result.numpy(), expected)
