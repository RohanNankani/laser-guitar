import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import datetime
import logging

def train_model(model_filename: str, NUM_EPOCHS: int, verbose=False):
    logging.basicConfig(level=(logging.INFO if verbose else logging.WARNING))
    if verbose: print("Tensorflow Version:", tf.__version__)

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        logging.warning('GPU device not found')
    logging.info('Found GPU at: {}'.format(device_name))

    data_dir = pathlib.Path("./finger_videos/training")

    image_count = len(list(data_dir.glob('*/*.png')))
    if verbose: print("Loaded a total of", image_count, "images")

    class_1 = list(data_dir.glob('class_1/*'))
    class_2 = list(data_dir.glob('class_2/*'))
    class_3 = list(data_dir.glob('class_3/*'))

    logging.info(f"Class 1: {len(class_1)} images")
    logging.info(f"Class 2: {len(class_2)} images")
    logging.info(f"Class 3: {len(class_3)} images")

    if verbose:
        print("Class 1 image:", str(class_1[0]))
        print("Class 2 image:", str(class_2[0]))
        print("Class 3 image:", str(class_3[0]))

    if verbose:
        plt.figure()
        plt.imshow(plt.imread(class_2[0]))
        plt.colorbar()
        plt.grid(False)
        plt.show()
        
        print("Sample image data:", plt.imread(class_2[0]))

    batch_size = 32
    img_height = 450
    img_width = 450

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=3525246402,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=2184260647,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    if verbose: print("List of class names:", str(class_names))

    if verbose:
        for image_batch, labels_batch in train_ds:
            print("Tensor shape for image batch:", image_batch.shape)
            print("Tensor shape for labels batch:", labels_batch.shape)
            break

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    model = tf.keras.Sequential([
        # tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3),
        tf.keras.layers.Dropout(rate=0.5),
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS,
        callbacks=[tensorboard_callback],
    )

    model.save(model_filename)

if __name__ == '__main__':
    train_model('v3_fingers.keras', verbose=True)