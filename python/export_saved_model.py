"""Script exporting a TF SavedModel computing Fibonacci numbers"""
import argparse
import pathlib
from typing import Dict

import tensorflow as tf

import fibonacci


class Model(tf.Module):
    """tf.Module representing TF SavedModel computing Fibonacci numbers."""
    @tf.function
    def __call__(self, tensor: tf.Tensor) -> Dict[str, tf.Tensor]:
        return dict(result=fibonacci.fibonacci_numbers(tensor))


def export_saved_model(path: pathlib.Path) -> None:
    """Write TF SavedModel to disk.

    Args:
        path: directory the SavedModel will be exported to
    """
    model = Model()

    call = model.__call__.get_concrete_function(
        tf.TensorSpec([
            None,
        ], tf.int64))  # pylint: disable=no-member
    tf.saved_model.save(model, str(path), signatures=call)


def run() -> None:
    """Run script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", required=True)
    args = parser.parse_args()

    export_saved_model(pathlib.Path(args.out_dir))


if __name__ == "__main__":
    run()
