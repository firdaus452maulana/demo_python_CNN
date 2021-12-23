import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


def do_data_preprocessing(train_dataset, validation_dataset, info,
                          BUFFER_SIZE, BATCH_SIZE):
    tokenizer = info.features["text"].encoder
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset_padded = train_dataset.padded_batch(
        BATCH_SIZE,
        tf.compat.v1.data.get_output_shapes(train_dataset)
    )
    validation_dataset_padded = validation_dataset.padded_batch(BATCH_SIZE,
                                                                tf.compat.v1.data.get_output_shapes(validation_dataset)
                                                                )
    return train_dataset_padded, validation_dataset_padded, tokenizer


def create_lstm_model(tokenizer):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.summary()
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model


def plot_history(train, val, title):
    epochs = range(len(train))
    plt.figure()
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="val")
    plt.title(title)
    plt.legend(loc="best")
    plt.pause(1.0)


def classify_reviews(test_reviews_path, model):
    with open(test_reviews_path, "r") as f:
        sample_reviews = f.read()
    sample_reviews = sample_reviews.split("\n")
    for review in sample_reviews:
        review_padded = tokenizer.encode(review)
    review_padded = tf.data.Dataset.from_tensors(review_padded)
    review_padded = review_padded.padded_batch(1)
    classes = model.predict(review_padded)
    print(f"classes[0]: {classes[0]}")
    if classes[0] > 0.5:
        print(f"[positive]\n{review[:100]}...")
    else:
        print(f"[negative]\n{review[:100]}...")


if __name__ == "__main__":
    print(tf.__version__)
    # Get the data
    dataset, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
    train_dataset, validation_dataset = dataset["train"], dataset["test"]
    BUFFER_SIZE = 10_000
    BATCH_SIZE = 64
    # Pre-process the dataset
    train_dataset_padded, validation_dataset_padded, tokenizer \
        = do_data_preprocessing(train_dataset, validation_dataset,
                                info, BUFFER_SIZE, BATCH_SIZE)
    # Build a LSTM model
    lstm_model = create_lstm_model(tokenizer)
    # Train the model to the training dataset
    history = lstm_model.fit(
        train_dataset_padded,
        epochs=10,
        validation_data=validation_dataset_padded
    )
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plot_history(acc, val_acc, "Training and validation accuracy")
    plot_history(loss, val_loss, "Training and validation loss")
    # Test sample reviews
    classify_reviews("datasets/imdb-sample-test.txt", lstm_model)
