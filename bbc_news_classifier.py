import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def read_js_stopword(filename):
    with open(filename, "r") as f:
        data = f.read()
    stopwords = data.split("\n")[-3]
    stopwords = [word[1:-1] for word in stopwords[10:-3].split(", ")]
    return stopwords


def read_test_text(file_path):
    with open(file_path, "r") as f:
        test_text = f.read()
    return test_text


def read_sample_text(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    test_text = []
    test_label = []
    for x in data:
        idx = x.find("];")
    test_text.append(x[idx + 2:])
    test_label.append(x[1:idx])
    return test_text, test_label


def do_data_preprocessing(sentences, label, training_portion, vocab_size,
                          oov_tok, padding_type, trunc_type, max_length):
    verify_feature = read_test_text("datasets/bbc-text-test.txt")
    print(f"sentences[0] == verify_feature => {sentences[0] == verify_feature}")
    train_size = int(len(sentences) * training_portion)
    train_sentences = sentences[:train_size]
    train_labels = label[:train_size]
    validation_sentences = sentences[train_size:]
    validation_labels = label[train_size:]
    # Fit sentences to create tokens
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index
    print(f"len(word_index): {len(word_index)}")
    # Tokenizing the sentences
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences,
                                 padding=padding_type,
                                 truncating=trunc_type,
                                 maxlen=max_length)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validation_sequences,
                                      padding=padding_type,
                                      truncating=trunc_type,
                                      maxlen=max_length)
    # Fit Tokenizer to sentences and create a sequence of token for labels
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(label)
    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
    return train_padded, training_label_seq, \
           validation_padded, validation_label_seq, \
           tokenizer, label_tokenizer


def create_language_model(vocab_size, embedding_dim, max_length):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(6, activation="softmax")  # zero index is for padding
    ])
    model.compile(loss="sparse_categorical_crossentropy",
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


def classify_news(news_arr, true_label_arr, model, tokenizer, label_tokenizer, max_length,
                  padding_type, trunc_type):
    reverse_label_index = dict([(v, k) for (k, v)
                                in label_tokenizer.word_index.items()])
    print(f"reverse_label_index: {reverse_label_index}")
    for i, news in enumerate(news_arr):
        test_sequence = tokenizer.texts_to_sequences([news])
    test_padded = pad_sequences(test_sequence, padding=padding_type,
                                maxlen=max_length, truncating=trunc_type)
    test_padded_numpy = np.array(test_padded)
    classes = model.predict(test_padded_numpy)
    get_class_key = np.argmax(classes[0])
    class_label = reverse_label_index[get_class_key]
    print(f"{news[:200]}: {class_label} [true label: {true_label_arr[i]}]")


if __name__ == "__main__":
    vocab_size = 10_000
    embedding_dim = 8
    max_length = 500
    trunc_type = "post"
    padding_type = "post"
    oov_tok = "<OOV>"
    training_portion = 0.8
    stopwords = read_js_stopword("datasets/stopwords.js")
    # Load text
    sentences = []
    label = []
    with open("datasets/bbc-text.csv", "r") as csvfile:
        datastore = csv.reader(csvfile)
    datastore.__next__()  # skip the first line
    for row in datastore:
        label.append(row[0])
    sentence = row[1]
    for word in stopwords:
        token = " " + word + " "
    sentence = sentence.replace(token, " ")
    sentences.append(sentence)
    # Data pre-processing
    train_padded, training_label_seq, \
    validation_padded, validation_label_seq, \
    tokenizer, label_tokenizer \
        = do_data_preprocessing(sentences, label, training_portion, vocab_size,
                                oov_tok, padding_type, trunc_type, max_length)
    # Build model
    lstm_model = create_language_model(vocab_size, embedding_dim, max_length)
    history = lstm_model.fit(
        train_padded,
        training_label_seq,
        epochs=50,
        validation_data=(validation_padded, validation_label_seq),
        verbose=1
    )
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plot_history(acc, val_acc, "Training and validation accuracy")
    plot_history(loss, val_loss, "Training and validation loss")
    # Test sample headlines
    news_arr, true_label_arr = read_sample_text("datasets/bbc-text-samples.txt")
    print(f"len(sample_sentences): {len(news_arr)}")
    classify_news(news_arr, true_label_arr, lstm_model, tokenizer, label_tokenizer,
                  max_length, padding_type, trunc_type)
