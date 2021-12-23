import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as keras_image


class MyCallback(tf.keras.callbacks.Callback):

    def __init__(self, desired_accuracy):
        super(MyCallback, self).__init__()
        self.DESIRED_ACCURACY = desired_accuracy


def on_epoch_end(self, epoch, logs=None):
    if logs.get("accuracy") > self.DESIRED_ACCURACY:
        print(f"\nReached {self.DESIRED_ACCURACY * 100:.2f}"
              + " accuracy so cancelling training!")
        self.model.stop_training = True


def load_dataset(zip_file_path, extracted_zip_file_path, train_happy_dir, train_sad_dir):
    zip_ref = zipfile.ZipFile(zip_file_path, "r")
    zip_ref.extractall(extracted_zip_file_path)
    zip_ref.close()

    train_happy_names = os.listdir(train_happy_dir)
    train_sad_names = os.listdir(train_sad_dir)

    print(train_happy_names[:10])
    print(train_sad_names[:10])
    print(f"total training happy images {len(train_happy_names)}")
    print(f"total training sad images {len(train_sad_names)}")

    return train_happy_names, train_sad_names


def do_data_preprocessing(dataset_dir):
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(150, 150),
        batch_size=8,  # reduce this from 128 (our dataset is small)
        class_mode="binary"
    )
    return train_generator


def create_cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(learning_rate=0.001),
                  metrics=["accuracy"])
    model.summary()
    return model


def plot_training_images(train_happy_dir, train_sad_dir,
                         train_happy_names, train_sad_names):
    # Parameter for our graph; we'll output images in a 4x4 configuration
    n_rows = 4
    n_cols = 4

    # Index for iterating over images
    img_index = 0

    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(n_cols * 4, n_rows * 4)

    img_index += 8
    next_happy_img = [os.path.join(train_happy_dir, fname)
                      for fname in train_happy_names[img_index - 8:img_index]]
    next_sad_img = [os.path.join(train_sad_dir, fname)
                    for fname in train_sad_names[img_index - 8:img_index]]
    for i, img_path in enumerate(next_happy_img + next_sad_img):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(n_rows, n_cols, i + 1)
        sp.axis("off")  # don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.pause(1.0)


def classify_images(fn_arr, model):
    for fn in fn_arr:
        path = "datasets/" + fn
        img = keras_image.load_img(path, target_size=(150, 150))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        image_i = np.vstack([x])
        classes = model.predict(image_i, batch_size=10)
        print(classes[0])
        if classes[0] > 0.5:
            print(fn + " is happy")
        else:
            print(fn + " is sad")


if __name__ == "__main__":
    zip_file_path = "datasets/happy-or-sad.zip"
    extracted_zip_file_path = "datasets/happy-or-sad"

    # Directory with our training happy images
    train_happy_dir = os.path.join("datasets/happy-or-sad/happy")

    # Directory with our training sad images
    train_sad_dir = os.path.join("datasets/happy-or-sad/sad")

    # Extract zip and get path to the data sets
    train_happy_names, train_sad_names = load_dataset(zip_file_path, extracted_zip_file_path,
                                                      train_happy_dir, train_sad_dir)

    # Plot the data sets
    plot_training_images(train_happy_dir, train_sad_dir, train_happy_names, train_sad_names)

    # Data preprocessing
    train_generator = do_data_preprocessing(extracted_zip_file_path)

    # Building a small CNN model
    cnn_model = create_cnn_model()

    # Training the model to the training data
    DESIRED_ACCURACY = 0.99
    callbacks = MyCallback(DESIRED_ACCURACY)
    history = cnn_model.fit(
        train_generator,
        steps_per_epoch=8,
        epochs=50,
        verbose=1,
        callbacks=[callbacks]
    )

    # Predict some images
    fn_arr = ["beauty-1132617_640.jpg", "girl-2961959_640.jpg",
              "woman-2126727_640.jpg", "beautiful-18279_640.jpg"]
    classify_images(fn_arr, cnn_model)
