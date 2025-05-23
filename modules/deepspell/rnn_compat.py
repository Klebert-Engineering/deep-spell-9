__all__ = ["MultiRNNCell"]
import tensorflow as tf


class MultiRNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """Simplified MultiRNNCell implementation for Keras 3."""

    def __init__(self, cells):
        super().__init__()
        if not isinstance(cells, (list, tuple)):
            raise TypeError("cells must be a list or tuple")
        self._cells = list(cells)

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size if self._cells else 0

    def zero_state(self, batch_size, dtype):
        return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)

    def call(self, inputs, state):
        output = inputs
        new_states = []
        for cell, s in zip(self._cells, state):
            output, new_s = cell(output, s)
            new_states.append(new_s)
        return output, tuple(new_states)
