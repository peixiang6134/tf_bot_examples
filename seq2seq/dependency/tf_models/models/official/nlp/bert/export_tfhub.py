# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A script to export the BERT core model as a TF-Hub SavedModel."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf
from typing import Optional, Text

from dependency.tf_models.models.official.nlp import bert_modeling
from dependency.tf_models.models.official.nlp import bert_models

FLAGS = flags.FLAGS

flags.DEFINE_string("bert_config_file", None,
                    "Bert configuration file to define core bert layers.")
flags.DEFINE_string("model_checkpoint_path", None,
                    "File path to TF model checkpoint.")
flags.DEFINE_string("export_path", None, "TF-Hub SavedModel destination path.")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("sp_model_file", None,
                    "The sentence piece model file that the ALBERT model was "
                    "trained on.")
flags.DEFINE_enum(
    "model_type", "bert", ["bert", "albert"],
    "Specifies the type of the model. "
    "If 'bert', will use canonical BERT; if 'albert', will use ALBERT model.")


def create_bert_model(bert_config: bert_modeling.BertConfig):
  """Creates a BERT keras core model from BERT configuration.

  Args:
    bert_config: A BertConfig` to create the core model.

  Returns:
    A keras model.
  """
  # Adds input layers just as placeholders.
  input_word_ids = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name="input_word_ids")
  input_mask = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name="input_mask")
  input_type_ids = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name="input_type_ids")
  transformer_encoder = bert_models.get_transformer_encoder(
      bert_config, sequence_length=None, float_dtype=tf.float32)
  sequence_output, pooled_output = transformer_encoder(
      [input_word_ids, input_mask, input_type_ids])
  # To keep consistent with legacy hub modules, the outputs are
  # "pooled_output" and "sequence_output".
  return tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[pooled_output, sequence_output]), transformer_encoder


def export_bert_tfhub(bert_config: bert_modeling.BertConfig,
                      model_checkpoint_path: Text,
                      hub_destination: Text,
                      vocab_file: Optional[Text] = None,
                      sp_model_file: Optional[Text] = None):
  """Restores a tf.keras.Model and saves for TF-Hub."""
  core_model, encoder = create_bert_model(bert_config)
  checkpoint = tf.train.Checkpoint(model=encoder)
  checkpoint.restore(model_checkpoint_path).assert_consumed()

  if isinstance(bert_config, bert_modeling.AlbertConfig):
    if not sp_model_file:
      raise ValueError("sp_model_file is required.")
    core_model.sp_model_file = tf.saved_model.Asset(sp_model_file)
  else:
    assert isinstance(bert_config, bert_modeling.BertConfig)
    if not vocab_file:
      raise ValueError("vocab_file is required.")
    core_model.vocab_file = tf.saved_model.Asset(vocab_file)
    core_model.do_lower_case = tf.Variable(
        "uncased" in vocab_file, trainable=False)
  core_model.save(hub_destination, include_optimizer=False, save_format="tf")


def main(_):
  assert tf.version.VERSION.startswith('2.')
  config_cls = {
      "bert": bert_modeling.BertConfig,
      "albert": bert_modeling.AlbertConfig,
  }
  bert_config = config_cls[FLAGS.model_type].from_json_file(
      FLAGS.bert_config_file)
  export_bert_tfhub(bert_config, FLAGS.model_checkpoint_path, FLAGS.export_path,
                    FLAGS.vocab_file, FLAGS.sp_model_file)


if __name__ == "__main__":
  app.run(main)
