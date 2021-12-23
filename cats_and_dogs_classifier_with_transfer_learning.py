import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
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
        batch_size=20,
        class_mode="binary",
        target_size=(150, 150)
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        batch_size=20,
        class_mode="binary",
        target_size=(150, 150)
    )
    return train_generator, validation_generator


def create_cnn_model(local_weights_file):
    pre_trained_model = VGG16(input_shape=(150, 150, 3),
                              include_top=False,
                              weights=None)
    pre_trained_model.load_weights(local_weights_file)
    for layer in pre_trained_model.layers:
        layer.trainable = False
    pre_trained_model.summary()
    last_layer = pre_trained_model.get_layer("block5_pool")
    print("last layer output shape: ", last_layer.output_shape)
    last_output = last_layer.output
    x = layers.Dropout(0.2)(last_output)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    model = Model(pre_trained_model.input, x)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
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
        else:
            print(fn + " is a cat")


if __name__ == "__main__":
    local_weights_file = "pre-trained/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    base_dir = "datasets/cats_and_dogs_filtered"
    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "validation")
    train_cats_dir = os.path.join(train_dir, "cats")
    train_dogs_dir = os.path.join(train_dir, "dogs")
    validation_cats_dir = os.path.join(validation_dir, "cats")
    validation_dogs_dir = os.path.join(validation_dir, "dogs")
    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)
    print(len(train_cat_fnames))
    print(len(train_dog_fnames))
    train_generator, validation_generator \
        = do_data_preprocessing(train_dir, validation_dir, aug=True)
    # Build a CNN model with pre-trained VGG16 net
    cnn_model = create_cnn_model(local_weights_file)
    history = cnn_model.fit(
        train_generator,
        # steps_per_epoch=100,
        epochs=20,
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
