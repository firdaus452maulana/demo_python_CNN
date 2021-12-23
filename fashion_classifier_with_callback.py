import tensorflow as tf


class MyCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") > 0.6:
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True


if __name__ == "__main__":
    mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    callbacks = MyCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    # will reach 60% accuracy in 2 epochs!
    model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    # Evaluate the model to the test data
    model.evaluate(x_train, y_train)
