"""Compute Fibonacci numbers in TensorFlow"""
import tensorflow as tf


def fibonacci_numbers(tensor: tf.Tensor) -> tf.Tensor:
    """Compute fibonacci numbers

    Args:
        tensor: tensor of dtype int32 or int64 and vector shape, i.e. [None,].
            For each entry n of `tensor` the n-th Fibonacci number will be
            computed.

    Return:
        tf.Tensor: same shape and dtype as `tensor`. Contains the Fibonacci
            numbers, see above.
    """
    max_index = tf.reduce_max(tensor)

    dtype = tensor.dtype
    i_init = tf.constant(1, dtype)
    z_init = tf.constant([0, 1], dtype)
    a_init = tf.constant(0, dtype)
    b_init = tf.constant(1, dtype)

    _, fib_numbers, _, _ = tf.while_loop(
        cond=lambda i, z, a, b: i < max_index,
        body=lambda i, z, a, b:
        [i + 1, tf.concat([z, [a + b]], axis=0), b, a + b],
        loop_vars=[i_init, z_init, a_init, b_init],
        shape_invariants=[
            i_init.get_shape(),
            tf.TensorSpec([None], dtype),
            a_init.get_shape(),
            b_init.get_shape()
        ])

    return tf.gather(fib_numbers, tensor)
