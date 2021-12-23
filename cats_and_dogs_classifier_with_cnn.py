import os
import zipfile
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.preprocessing import image as keras_image


def load_dataset(zip_file_path, extracted_zip_file_path):
    zip_ref = zipfile.ZipFile(zip_file_path, "r")
    zip_ref.extractall(os.path.split(extracted_zip_file_path)[0])
    zip_ref.close()

    train_dir = os.path.join(extracted_zip_file_path, "train")
    validation_dir = os.path.join(extracted_zip_file_path, "validation")

    train_cats_dir = os.path.join(train_dir, "cats")
    train_dogs_dir = os.path.join(train_dir, "dogs")

    validation_cats_dir = os.path.join(validation_dir, "cats")
    validation_dogs_dir = os.path.join(validation_dir, "dogs")

    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)

    validation_cat_fnames = os.listdir(validation_cats_dir)
    validation_dog_fnames = os.listdir(validation_dogs_dir)

    print(train_cat_fnames[:10])
    print(train_dog_fnames[:10])

    print("total training cat images :", len(train_cat_fnames))
    print("total training dog images :", len(train_dog_fnames))

    print("total validation cat images:", len(validation_cat_fnames))
    print("total validation dog images:", len(validation_dog_fnames))

    return train_dir, validation_dir, train_cats_dir, train_dogs_dir, \
           train_cat_fnames, train_dog_fnames


def do_data_preprocessing(train_dir, validation_dir):
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


def create_cnn_model():
    # Building a small model from scratch: validation accuracy ~ 72 %
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
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def plot_cats_and_dogs(train_cats_dir, train_dogs_dir,
                       train_cat_fnames, train_dog_fnames):
    # Parameter for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4
    # Index for iterating over images
    pic_index = 0
    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)
    pic_index += 8
    next_cat_pix = [os.path.join(train_cats_dir, fname)
                    for fname in train_cat_fnames[pic_index - 8:pic_index]]
    next_dog_pix = [os.path.join(train_dogs_dir, fname)
                    for fname in train_dog_fnames[pic_index - 8:pic_index]]
    for i, img_path in enumerate(next_cat_pix + next_dog_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis("off")  # Don't show axes (or gridlines)
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
            print(fn + " is a dog")
        else:
            print(fn + " is a cat")


def plot_intermediate_repr(model, train_cats_dir, train_dogs_dir,
                           train_cat_fnames, train_dog_fnames):
    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after the first.
    successive_outputs = [layer.output for layer in model.layers[0:]]
    visualization_model = tf.keras.models.Model(inputs=model.input,
                                                outputs=successive_outputs)
    # Let's prepare a random input image from the training set.
    cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
    dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]
    img_path = random.choice(cat_img_files + dog_img_files)
    img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
    x = img_to_array(img)  # numpy array with shape (150, 150, 3)
    print(f"x.shape : {x.shape}")
    x = x.reshape((1,) + x.shape)
    x = x / 255.
    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)
    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]
    # Now let's display our representations
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        # Just do this for the conv / maxpool layers, not the fully connected layers
        if len(feature_map.shape) == 4:
            n_features = feature_map.shape[-1]  # number of features (filters) in feature map
            # The feature map has shape (1, size, size, n_features)
            size = feature_map.shape[1]
            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
                # Post-process the feature to make it visually palatable
                x = feature_map[0, :, :, i]
                x -= x.mean()
                # print('x.std()', x.std())
                x = x / x.std() if x.std() > 1e-14 else x
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype("uint8")
                # We will tile each filter into this big horizontal grid
                display_grid[:, i * size: (i + 1) * size] = x
            # Display the grid
            plt.figure(figsize=(20, 2))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect="auto", cmap="viridis")
            plt.subplots_adjust(left=0.03, right=0.99)
    plt.pause(1.0)


def plot_history(train, val, title):
    epochs = range(len(train))
    plt.figure()
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="val")
    plt.title(title)
    plt.legend(loc="best")
    plt.pause(1.0)


if __name__ == "__main__":
    zip_file_path = "datasets/cats_and_dogs_filtered.zip"
    extracted_zip_file_path = "datasets/cats_and_dogs_filtered"
    # Extract zip and get path to the data sets
    train_dir, validation_dir, train_cats_dir, train_dogs_dir, train_cat_fnames, train_dog_fnames = load_dataset(
        zip_file_path, extracted_zip_file_path)
    # Plot the data sets
    plot_cats_and_dogs(train_cats_dir, train_dogs_dir,
                       train_cat_fnames, train_dog_fnames)
    # Data preprocessing
    train_generator, validation_generator = do_data_preprocessing(train_dir, validation_dir)
    # Building a small CNN model
    cnn_model = create_cnn_model()
    history = cnn_model.fit(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=100,
        epochs=15,
        validation_steps=50,
        verbose=1
    )
    fn_arr = ["cat-2083492_only_head.jpg", "cat-1146504_640.jpg",
              "dog-3846767_640.jpg", "dog-3388069_640.jpg"]
    classify_images(fn_arr, cnn_model)
    # Visualizing intermediate representations
    plot_intermediate_repr(cnn_model, train_cats_dir, train_dogs_dir,
                           train_cat_fnames, train_dog_fnames)
    # Evaluating accuracy and loss for the model
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plot_history(acc, val_acc, "Training and validation accuracy")
    plot_history(loss, val_loss, "Training and validation loss")
