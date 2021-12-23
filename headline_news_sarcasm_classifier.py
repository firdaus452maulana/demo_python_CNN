import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def do_data_preprocessing(dataset_path, vocab_size, max_length,
                          padding_type, trunc_type):
    oov_tok = "<OOV>"  # out of vocabulary token
    training_size = 20_000  # in total there are 26,709 items

    with open(dataset_path, "r") as f:
        datastore = json.load(f)

    sentences = []
    labels = []

    for item in datastore:
        sentences.append(item["headline"])
    labels.append(item["is_sarcastic"])
    training_sentences = sentences[:training_size]
    validation_sentences = sentences[training_size:]
    training_labels = labels[:training_size]
    validation_labels = labels[training_size:]
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    print(word_index)
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences,
                                    maxlen=max_length,
                                    padding=padding_type,
                                    truncating=trunc_type)
    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validation_sequences,
                                      maxlen=max_length,
                                      padding=padding_type,
                                      truncating=trunc_type)
    validation_padded = np.array(validation_padded)
    validation_labels = np.array(validation_labels)
    return training_padded, training_labels, \
           validation_padded, validation_labels, tokenizer


def create_lstm_model(vocab_size, embedding_dim, max_length):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(learning_rate=0.001),
                  metrics=["accuracy"])
    model.summary()

    return model


def plot_history(train, val, title):
    epochs = range(len(train))
    plt.figure()
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="val")
    plt.title(title)
    plt.legend(loc="best")
    plt.pause(1.0)


def classify_headlines(headline_arr, model, tokenizer, max_length,
                       padding_type, trunc_type):
    for headline in headline_arr:
        test_sequence = tokenizer.texts_to_sequences(headline)
    test_padded = pad_sequences(test_sequence,
                                maxlen=max_length,
                                padding=padding_type,
                                truncating=trunc_type)
    test_padded = np.array(test_padded)
    classes = model.predict(test_padded)
    print(classes[0])
    if classes[0] > 0.5:
        print(headline + ": a sarcasm")
    else:
        print(headline + ": not a sarcasm")


if __name__ == "__main__":
    print(tf.__version__)
    
    dataset_path = "datasets/sarcasm.json"
    vocab_size = 1_000
    embedding_dim = 16
    max_length = 120

    trunc_type = "post"
    padding_type = "post"

    training_padded, training_labels, validation_padded, validation_labels, \
    tokenizer = do_data_preprocessing(dataset_path, vocab_size, max_length,
                                      trunc_type, padding_type)
    lstm_model = create_lstm_model(vocab_size, embedding_dim, max_length)

    history = lstm_model.fit(
        training_padded,
        training_labels,
        epochs=100,
        validation_data=(validation_padded, validation_labels),
        verbose=1
    )

    lstm_model.save("model-saved/categ4.h5")
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plot_history(acc, val_acc, "Training and validation accuracy")
    plot_history(loss, val_loss, "Training and validation loss")

    # Test sample headlines
    headline_arr = ["Tokyo's COVID-19 cases hit all-time high despite state of emergency",
                    "China: No need for 2nd WHO probe on virus origin",
                    "Turkey builds border wall to block Afghan migrants",
                    "Line accounts of Taiwan officials hacked"]
    classify_headlines(headline_arr, lstm_model, tokenizer, max_length,
                       padding_type, trunc_type)
