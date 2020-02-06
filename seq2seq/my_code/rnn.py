import dependency.opennmt_v2.layers.rnn as rnn
from dependency.opennmt_v2.layers.reducer import ConcatReducer
import json
from my_code.util.format_check_util import to_rnn_type
import tensorflow as tf


class MyRNN(tf.keras.layers.Layer):
    """wrapper of RNN, providing dropout, residual connections and state reducer for 3 types of RNN"""

    def __init__(self, json_str, training):
        """
        initialize MyRNN
        :param json_str: json string to config MyRNN. the corresponding json contains the following fields
            num_layers: int, number of RNN layers
            num_units: int, number of units in each RNN layer
            dropout: float, input dropout probability for each layer
            residual_connections: bool, if having residualconnection for each layer
            cell_class: string, LSTM, GRU, Vanilla (or Simple)
            bidirectional: bool, if having bidirection RNN
            sequence_or_states: string, "sequence_only", "states_only", "both"
        :param training:
        """
        super(MyRNN, self).__init__()
        self.config = json.loads(json_str)
        self.training = training
        self.cell = rnn.make_rnn_cell(
            self.config["num_layers"],
            self.config["num_units"],
            self.config["dropout"],
            self.config["residual_connections"],
            to_rnn_type(self.config["cell_class"]))
        self.rnn = rnn.RNN(self.cell, self.config["bidirectional"], ConcatReducer())

    def call(self, inputs, mask=None, initial_state=None):
        """
        run RNN
        :param inputs: tensor list of size 1. the shape of the tensor is [batch_size, sequence_length, embedding_size]
        :return: tensor list of size 2.
            if bidrection, sequence tensor's shape [batch_size, sequence_length, 2 * num_units]
                           state tensor's shape [batch_size, num_layers, 2 * num_units]
            otherwise: sequence tensor's shape [batch_size, sequence_length, num_units]
                       state tensor's shape [batch_size, num_layers, num_units]
        """
        output_tensor, state_tensor = self.rnn(inputs, mask=mask, training=self.training, initial_state=initial_state)
        if self.config["sequence_or_states"] == "sequence_only":
            return output_tensor
        elif self.config["sequence_or_states"] == "states_only":
            return state_tensor
        else:
            return (output_tensor,) + state_tensor
