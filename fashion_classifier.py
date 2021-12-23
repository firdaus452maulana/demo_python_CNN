### Classifying Fashion MNIST ###
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

if __name__ == "__main__":
    # Load the dataset
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    # Plot an image from the dataset
    plt.imshow(training_images[0], cmap="Greys")
    plt.pause(1)  # increase the pause time to give time for plotting
    print(training_labels[0])
    print(training_images[0])
    # Normalizing image intensities to [0, 1]
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    # Create a model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    # Compile the model and train it to the dataset
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(training_images, training_labels, epochs=5)
    # Evaluate the model to the test data
    model.evaluate(test_images, test_labels)
