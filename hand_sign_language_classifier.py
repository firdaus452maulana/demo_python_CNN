import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf


def get_data(filename):
    with open(filename) as training_file:
        _ = training_file.readline()  # skip first line
        data = training_file.readlines()
        labels = []
        images = []
        num_of_data = len(data)
        for i, row in enumerate(data):
            row = row.strip("\n").split(",")
            labels.append(row[0])
            images.append(np.array_split(row[1:785], 28))
            sys.stdout.write(f"\rprocessing: {(i + 1) / float(num_of_data) * 100:.2f} %")
            sys.stdout.flush()
        print("")
        labels = np.array(labels).astype(float)
        images = np.array(images).astype(float)
    return images, labels


def plot_one_image(image_data, image_label):
    fig, ax = plt.subplots()
    ax.imshow(image_data, cmap="gray", vmin=0, vmax=255)
    num_to_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                       'T', 'U', 'V', 'W', 'X', 'Y']
    ax.set_title("image_label = {:g} ({:s})".format(image_label,
                                                    num_to_alphabet[int(image_label)]))
    plt.pause(1.0)


def do_data_preprocessing(training_images, training_labels,
                          validation_images, validation_labels):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode="nearest"
    )
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    training_generator = train_datagen.flow(
        training_images,
        training_labels,
        batch_size=20,
    )
    validation_generator = validation_datagen.flow(
        validation_images,
        validation_labels,
        batch_size=20
    )
    return training_generator, validation_generator


def create_cnn_model():
    # image size is 28x28, we don't need third conv2d
    # This will make the image size 1x1px!
    # second conv2D will make the image size 5x5px
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        # tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        # tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(26, activation="softmax")  # labels have value 0 - 24
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
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


def classify_images(fn_arr, model):
    for fn in fn_arr:
        path = "datasets/" + fn
    # because we train using grayscale image, we need to convert
    # the sample image using color_mode="grayscale"
    img = keras_image.load_img(path, target_size=(28, 28),
                               color_mode="grayscale")
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    num_to_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                       'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                       'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    num_to_alphabet = np.array(num_to_alphabet)
    image_i = np.vstack([x])
    classes = model.predict(image_i, batch_size=10)
    print(classes[0])
    class_label = num_to_alphabet[classes[0].astype(np.int32) > 0.5][0]
    print(fn + " is a letter {:s}".format(class_label))


if __name__ == "__main__":
    # Load the dataset
    training_images, training_labels \
        = get_data(os.getcwd() + "/datasets/sign_mnist_train.csv")
    validation_images, validation_labels \
        = get_data(os.getcwd() + "/datasets/sign_mnist_test.csv")
    print(training_images.shape)
    print(training_labels.shape)
    print(validation_images.shape)
    print(validation_labels.shape)
    training_images = np.expand_dims(training_images, axis=-1)
    validation_images = np.expand_dims(validation_images, axis=-1)
    print(training_images.shape)
    print(validation_images.shape)
    print(np.max(validation_labels), np.min(validation_labels))
    print(np.max(training_labels), np.min(training_labels))
    # Plot one of the images and its label
    image_num = 2
    plot_one_image(training_images[image_num, :, :, 0],
                   training_labels[image_num])
    # Data pre-preprocessing with ImageDataGenerator
    training_generator, validation_generator \
        = do_data_preprocessing(training_images, training_labels,
                                validation_images, validation_labels)
    # Build a CNN model
    cnn_model = create_cnn_model()
    history = cnn_model.fit(
        training_generator,
        # steps_per_epoch=len(training_images)/20,
        epochs=50,
        validation_data=validation_generator,
        # validation_steps=len(validation_images)/20
    )
    # Evaluate the model on the validation dataset
    cnn_model.evaluate(validation_generator)
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plot_history(acc, val_acc, "Training and validation accuracy")
    plot_history(loss, val_loss, "Training and validation loss")
    # Test on sample images
    fn_arr = ["alphabet-letter-C-1298289_640.png",
              "alphabet-letter-D-1298315_640.png",
              "alphabet-letter-Y-1298311_640.png",
              "sign-language-letter-A-28717_640.png"]
    classify_images(fn_arr, cnn_model)
