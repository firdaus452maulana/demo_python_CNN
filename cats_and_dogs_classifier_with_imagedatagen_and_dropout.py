import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as keras_image


def plot_history(train, val, title):
    epochs = range(len(train))
    plt.figure()
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="val")
    plt.title(title)
    plt.legend(loc="best")
    plt.pause(1.0)


def do_data_preprocessing(train_dir, validation_dir, aug=False):
    if aug:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode="binary"
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode="binary"
    )
    return train_generator, validation_generator


def create_cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),  # 3rd options (with image augmentation)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(lr=1e-4),
                  metrics=["accuracy"])
    model.summary()
    return model


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
            print(fn + " is a dog")
        else: print(fn + " is a cat")

if __name__ == "__main__":
    base_dir = "datasets/cats_and_dogs_filtered"
    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "validation")
    train_cats_dir = os.path.join(train_dir, "cats")
    train_dogs_dir = os.path.join(train_dir, "dogs")
    validation_cats_dir = os.path.join(train_dir, "dogs")
    validation_dogs_dir = os.path.join(validation_dir, "cats")
    print(len(os.listdir(train_cats_dir)))
    print(len(os.listdir(train_dogs_dir)))
    print(len(os.listdir(validation_cats_dir)))
    print(len(os.listdir(validation_dogs_dir)))
    # 1st options without image augmentation and dropout layer
    # 2nd options with image augmentation without dropout layer
    # 3rd options with image augmentation and dropout layer
    train_generator, validation_generator \
        = do_data_preprocessing(train_dir, validation_dir, aug=True)
    # Build a CNN model
    cnn_model = create_cnn_model()
    history = cnn_model.fit(
        train_generator,
        # steps_per_epoch=100, # 2000 images = batch_size * steps
        epochs=100,
        validation_data=validation_generator,
        # validation_steps=50,
        verbose=1
    )
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plot_history(acc, val_acc, "Training and validation accuracy")
    plot_history(loss, val_loss, "Training and validation loss")
    # test the model
    fn_arr = ["cat-2083492_only_head.jpg", "cat-1146504_640.jpg",
              "dog-3846767_640.jpg", "dog-3388069_640.jpg"]
    classify_images(fn_arr, cnn_model)



