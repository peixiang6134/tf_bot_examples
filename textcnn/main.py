import tensorflow as tf
from my_code.data_loader import train_dataset, dev_dataset, test_dataset
from my_code.word_embedding import MyWordEmbedding
from tensorflow.python.keras.layers import Lambda
from my_code.cnn_4_ngram import MyCNN4Ngram
from tensorflow import keras
from my_code.time_step_mask_computation import compute_2d_mask_padding_token
from my_code.time_step_mask_computation import run_mask_embedding


def generator_for_train():
    dataset_iter, steps_per_epoch = train_dataset()

    def batch_iter():
        for features, targets in dataset_iter():
            t = [tf.keras.backend.one_hot(tf.strings.to_number(targets["label"], tf.int32), 6)]
            f = [features["text"]]
            yield f, [t[0]]
    return batch_iter(), steps_per_epoch


def generator_for_dev():
    dataset_iter, steps_per_epoch = dev_dataset()

    def batch_iter():
        for features, targets in dataset_iter():
            t = [tf.keras.backend.one_hot(tf.strings.to_number(targets["label"], tf.int32), 6)]
            f = [features["text"]]
            yield f, [t[0]]
    return batch_iter(), steps_per_epoch


def generator_for_test():
    dataset_iter, steps_per_epoch = test_dataset()

    def batch_iter():
        for features, targets in dataset_iter():
            f = [features["text"]]
            yield f
    return batch_iter(), steps_per_epoch


def get_model(training):
    json_str = '{"vocabulary_file_name": "./voc.txt", "case_insensitive_embeddings": false, "num_oov_buckets": 1, "word_embedding_size": 128, "voc_size": 20000, "word_embedding_file_name": "embedding.txt", "with_header": false, "trainable": true}'
    word_emb_op = MyWordEmbedding(json_str)
    json_str = '{"num_outputs": 256, "kernel_size": 5, "strides": 1, "activation": null}'
    cnn_op = MyCNN4Ngram(json_str, training)
    dense_op = tf.keras.layers.Dense(units=6, kernel_initializer="random_uniform", bias_initializer="zeros", activation="softmax", use_bias=False)
    text = tf.keras.Input([128], dtype=tf.string)
    word_emb_out = word_emb_op(text)  # [128, 128, 128]
    text_mask = compute_2d_mask_padding_token(text, "<blank>")
    word_emb_out = run_mask_embedding(word_emb_out, text_mask)
    cnn_out = cnn_op(word_emb_out)  # [128, 128, 128]
    cnn_out = run_mask_embedding(cnn_out, text_mask)
    pool_out = keras.layers.GlobalMaxPooling1D(data_format='channels_last')(cnn_out)  # [128, 128]
    dense_out = dense_op(pool_out)  # [128, 6]
    model = tf.keras.Model(inputs=[text], outputs=[dense_out])
    model.summary()
    # tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, expand_nested=True)

    return model


if __name__ == "__main__":
    train_iter, train_steps = generator_for_train()
    dev_iter, dev_steps = generator_for_dev()
    test_iter, test_steps = generator_for_test()

    training = True
    prediction = True

    model = get_model(training=training)
    if training:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        model.compile(optimizer=optimizer, loss=[tf.keras.losses.categorical_crossentropy], loss_weights=[1.0])
        model.fit_generator(generator=train_iter, steps_per_epoch=train_steps*1, epochs=1, validation_data=dev_iter, validation_steps=dev_steps*1)
        model.save_weights("model_best.h5")

    if prediction:
        import numpy as np
        model.load_weights("model_best.h5", by_name=True)
        result = model.predict(x=test_iter)
        print("result:", result, "length:", len(result))
        labels = [int(l.strip()) for l in open("./data/processed/test.tgt").readlines()]
        right = 0
        for i, lab in enumerate(labels):
            r = result[i]
            pred = np.argmax(r)
            if pred == lab:
                right += 1
        acc = right * 1.0 / len(labels)
        print("accuracy:", acc)

