import tensorflow as tf
from my_code.data_loader import train_dataset, dev_dataset, test_dataset
import numpy as np
import tensorflow as tf
from my_code.word_embedding import MyWordEmbedding
from my_code.time_step_mask_computation import compute_2d_mask_padding_token
from dependency.tf_models.models.official.transformer.v2 import beam_search
from my_code.rnn import MyRNN


def get_voc():
    token2id = {}
    id2token = {}
    input = open("./data/voc.txt", "r", encoding="utf8").readlines()
    for i, item in enumerate(input):
        token = item.strip()
        token2id[token] = i
        id2token[i] = token
    length = len(token2id)
    token2id["<unk>"] = length
    id2token[length] = "<unk>"
    return token2id, id2token


def generator_for_train():
    dataset_iter, steps_per_epoch = train_dataset()

    def batch_iter():
        for features, targets in dataset_iter():
            t_out = tf.strings.to_number(targets["l0"], tf.int32)
            t_in = tf.pad(t_out[:, :-1], [[0, 0], [1, 0]], constant_values=1)
            f = [features["f0"], t_in]
            yield f, t_out
    return batch_iter(), steps_per_epoch


def generator_for_dev():
    dataset_iter, steps_per_epoch = dev_dataset()

    def batch_iter():
        for features, targets in dataset_iter():
            t_out = tf.strings.to_number(targets["l0"], tf.int32)
            t_in = tf.pad(t_out[:, :-1], [[0, 0], [1, 0]], constant_values=1)
            f = [features["f0"], t_in]
            yield f, t_out
    return batch_iter(), steps_per_epoch


def generator_for_test():
    dataset_iter, steps_per_epoch = test_dataset()

    def batch_iter():
        for features, targets in dataset_iter():
            yield features["f0"]
    return batch_iter(), steps_per_epoch


class Encoder(tf.keras.layers.Layer):
    def __init__(self, training):
        super(Encoder, self).__init__()
        json_str = '{"is_ids": false, "vocabulary_file_name": "./data/voc.txt", "case_insensitive_embeddings": false, "num_oov_buckets": 1, "word_embedding_size": 128, "voc_size": 6511, "word_embedding_file_name": "embedding.txt", "with_header": false, "trainable": true}'
        self.we = MyWordEmbedding(json_str)
        json_str = '{"initial_state": false, "num_layers": 2, "num_units": 128, "dropout": 0.5, "residual_connections": true, "cell_class": "LSTM", "bidirectional": true, "sequence_or_states": "states_only"}'
        self.rnn = MyRNN(json_str, training)

    def call(self, f0):
        emb_out = self.we(f0)
        mask_out = compute_2d_mask_padding_token(f0, "<blank>")
        _, encoder_outputs = self.rnn(emb_out, mask_out)
        return encoder_outputs


class Decoder(tf.keras.layers.Layer):
    def __init__(self, training):
        super(Decoder, self).__init__()
        json_str = '{"is_ids": true, "vocabulary_file_name": "./data/voc.txt", "case_insensitive_embeddings": false, "num_oov_buckets": 1, "word_embedding_size": 128, "voc_size": 6511, "word_embedding_file_name": "embedding.txt", "with_header": false, "trainable": true}'
        self.we = MyWordEmbedding(json_str)
        json_str = '{"initial_state": true, "num_layers": 1, "num_units": 256, "dropout": 0.5, "residual_connections": true, "cell_class": "LSTM", "bidirectional": false, "sequence_or_states": "both"}'
        self.rnn = MyRNN(json_str, training)
        self.dense = tf.keras.layers.Dense(units=6512, kernel_initializer="random_uniform", bias_initializer="zeros", activation="softmax", use_bias=False)
        if training is False:
            self.vocab_size = 6512
            self.beam_size = 2
            self.alpha = 0.8
            self.eos_id = 2

    def call(self, target, encoder_outputs):
        state_h, state_c = encoder_outputs
        states = ([state_h, state_c],)
        emb_out = self.we(target)
        target_mask = compute_2d_mask_padding_token(target, 0)
        rnn_out, decoder_states = self.rnn(emb_out, target_mask, states)
        softmax_out = self.dense(rnn_out)
        return softmax_out, decoder_states

    def _get_symbols_to_logits_fn(self):

        def symbols_to_logits_fn(ids, i, cache):
            decoder_input = ids[:, -1:]
            encoder_outputs = cache["encoder_outputs"]
            softmax_out, decoder_states = self.call(decoder_input, encoder_outputs)
            cache["encoder_outputs"] = decoder_states
            softmax_out = tf.squeeze(softmax_out, axis=[1])
            return softmax_out, cache

        return symbols_to_logits_fn

    def predict(self, encoder_outputs):
        batch_size = tf.shape(encoder_outputs[0])[0]
        input_length = tf.shape(encoder_outputs[0])[1] #TODO: this is the hidden_states dim, not the input_length
        max_decode_length = 20
        symbols_to_logits_fn = self._get_symbols_to_logits_fn()
        initial_ids = tf.ones([batch_size], dtype=tf.int32)
        cache = {}
        cache["encoder_outputs"] = encoder_outputs

        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.vocab_size,
            beam_size=self.beam_size,
            alpha=self.alpha,
            max_decode_length=max_decode_length,
            eos_id=self.eos_id
        )

        # return all the beams
        top_decoded_ids = decoded_ids[:, :1, 1:]
        top_scores = scores[:, :1]
        return {"outputs": top_decoded_ids, "scores": top_scores}


class Seq2Seq(tf.keras.Model):
    def __init__(self, training, name=None):
        super(Seq2Seq, self).__init__(name)
        self.encoder = Encoder(training)
        self.decoder = Decoder(training)

    def call(self, inputs):
        if len(inputs) == 2:
            inputs, targets = inputs[0], inputs[1]
        else:
            inputs, targets = inputs[0], None

        encoder_outputs = self.encoder(inputs)
        if targets is None:
            return self.decoder.predict(encoder_outputs)
        else:
            softmax_out, _ = self.decoder(targets, encoder_outputs)
            return softmax_out


def create_model(training):
    if training:
        inputs = tf.keras.Input([None], dtype=tf.string)
        targets = tf.keras.Input([None], dtype=tf.int32)
        internal_model = Seq2Seq(training, name="Seq2Seq")
        softmax_out = internal_model([inputs, targets])
        model = tf.keras.Model([inputs, targets], softmax_out)
        return model
    else:
        inputs = tf.keras.Input([None], dtype=tf.string)
        #inputs = np.array([["a", "b", "c"]])
        internal_model = Seq2Seq(training, name="Seq2Seq")
        ret = internal_model([inputs])
        outputs, scores = ret["outputs"], ret["scores"]
        print(outputs, scores)
        model = tf.keras.Model(inputs, [outputs, scores])
        model.summary()
        return model


def test_model():
    model = create_model(True)
    model.load_weights("checkpoints/weights.01-0.32")
    input = open("./data/debug_src.txt", "r", encoding="utf8").readlines()
    input = np.array([input[0].strip().split()])
    target = np.array([[1]])
    result = model.predict([input, target])
    ids = np.argmax(result)
    score = result[0,0, ids]
    print("test training model ok.")


def test_parameter():
    inputs = tf.keras.Input([None], dtype=tf.string)
    encoder = Encoder(training)
    output = encoder(inputs)
    encoder_model = tf.keras.Model(inputs, output)
    encoder_model.summary()
    print("test encoder model ok.")

    target = tf.keras.Input([None], dtype=tf.int32)
    state_h = tf.keras.Input([None], dtype=tf.float32)
    state_c = tf.keras.Input([None], dtype=tf.float32)

    decoder = Decoder(training)
    dec_out = decoder(target, [state_h, state_c])
    decoder_model = tf.keras.Model([target, state_h, state_c], dec_out)
    decoder_model.summary()
    print("test decoder model ok.")

if __name__ == "__main__":
    train_iter, train_steps = generator_for_train()
    dev_iter, dev_steps = generator_for_dev()
    test_iter, test_steps = generator_for_test()

    # training = False
    training = True
    if training:
        model = create_model(training)
        model.summary()
        #tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, expand_nested=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        model.compile(optimizer=optimizer, loss=[tf.keras.losses.sparse_categorical_crossentropy], loss_weights=[1.0])
        callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/weights.{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="auto"
        )
        model.fit(x=train_iter, steps_per_epoch=train_steps*10, epochs=1,
                  validation_data=dev_iter, validation_steps=dev_steps*10,
                  callbacks=[callback])
    else: # inference/predict
        test_parameter()
        test_model()
        token2id, id2token = get_voc()
        tf.random.set_seed(1234)
        np.random.seed(1234)
        token2id, id2token = get_voc()
        model = create_model(training)
        model.load_weights("checkpoints/weights.01-0.32")
        print("load_weights ok")
        result = model.predict(x=test_iter)
        #print("result:", result, "length:", len(result))
        outputs = result[0]
        scores = result[1]
        for i in range(len(outputs)):
            output = outputs[i]
            score = scores[i]
            print("{}-th sample:".format(i))
            for j, out in enumerate(output):
                sentence = [id2token[t] for t in out if t in id2token and t not in [0,1,2]]
                sentence = "".join(sentence)
                s = score[j]
                print("sentence-{}:{}, score:{}".format(j, sentence, s))
        print("ok")
