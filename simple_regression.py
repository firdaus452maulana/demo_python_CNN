import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


def plot_data(x_data, y_data):
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data, 'ro')
    plt.pause(1)
    return None


if __name__ == "__main__":
    # Define and compile the neural networks
    model = tf.keras.Sequential(
        [keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer="sgd", loss="mean_squared_error")
    # Providing the data
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])  # y = 2x - 1
    # Plotting data will help us to understand faster
    # visually about the data
    plot_data(xs, ys)
    # Train the neural network
    model.fit(xs, ys, epochs=500)
    # Predict using the model
    print(model.predict(([10.0])))  # this result should be close to 19.