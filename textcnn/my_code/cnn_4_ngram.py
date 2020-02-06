import tensorflow as tf
import json


class MyCNN4Ngram:
    """
    use 1D CNN for ngram modeling
    """
    def __init__(self, json_str, *args):
        """
        initialize MyCNN4Ngram with a json string
        :param json_str: string, corresponding json containing the following fields:
            num_outputs: number of CNN filters
            kernel_size: int, CNN kernel size
            strides: int, specifying the stride length of the convolution
            activation: string, Activation function to use. f you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        """
        self.config = json.loads(json_str)
        self.conv1D = tf.keras.layers.Conv1D(
            self.config["num_outputs"],
            self.config["kernel_size"],
            self.config["strides"],
            'valid',
            'channels_last',
            1,
            self.config["activation"])

    def _run_cnn(self, inputs):
        inputs = tf.pad(inputs, [[0, 0], [0, self.config["kernel_size"] - 1], [0, 0]])
        outputs = self.conv1D(inputs)
        return outputs

    def __call__(self, inputs):
        """to keep the shape unchanged, add a few padding first"""
        return self._run_cnn(inputs)
