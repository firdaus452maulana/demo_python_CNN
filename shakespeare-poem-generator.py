import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.utils as keras_utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import regularizers


def do_data_preprocessing(data_path):
    with open(data_path, "r") as f:
        data = f.read()
    corpus = data.lower().split("\n")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    # create input sequence using list of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
    input_sequences.append(n_gram_sequence)
    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,
                                             maxlen=max_sequence_len,
                                             padding="pre"))
    # create predictors and label
    predictors, labels = input_sequences[:, :-1], input_sequences[:, -1]
    label_categ = keras_utils.to_categorical(labels, num_classes=total_words)
    return predictors, label_categ, total_words, max_sequence_len, tokenizer


def create_language_model(total_words, ):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len - 1),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(total_words / 2, activation="relu",
                              kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.Dense(total_words, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(learning_rate=0.001),
                  metrics=["accuracy"])
    model.summary()
    return model


def plot_history(train, title):
    epochs = range(len(train))
    plt.figure()
    plt.plot(epochs, train)
    plt.title(title)
    plt.pause(1.0)


def generate_text(sample_text, next_words, tokenizer, max_sequence_len, model):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([sample_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    predicted = np.argmax(model.predict(token_list, verbose=1), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    sample_text += " " + output_word
    print(sample_text)


if __name__ == "__main__":
    # data_path = "datasets/chairil-anwar-poems.txt"
    data_path = "datasets/sonnets.txt"
    predictors, label_categ, total_words, \
    max_sequence_len, tokenizer = do_data_preprocessing(data_path)
    # Build a language model
    lstm_model = create_language_model(total_words)
    history = lstm_model.fit(predictors, label_categ, epochs=100, verbose=1)
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    plot_history(acc, "Training accuracy")
    plot_history(loss, "Training loss")
    sample_text = "Help me Obi Wan Kenobi, you're my only hope"
    # sample_text = "saat kenyataan hanyalah bayangan dan semu" # indonesian words
    next_words = 100
    generate_text(sample_text, next_words, tokenizer, max_sequence_len, lstm_model)
