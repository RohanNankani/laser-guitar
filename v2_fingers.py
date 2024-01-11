import tensorflow as tf
import numpy as np
import pathlib

print("Tensorflow Version:", tf.__version__)

data_dir = pathlib.Path("./finger_videos/training")

image_count = len(list(data_dir.glob('*/*.png')))
print("Loaded a total of", image_count, "images")

class_1 = list(data_dir.glob('class_1/*'))
class_2 = list(data_dir.glob('class_2/*'))
class_3 = list(data_dir.glob('class_3/*'))

print(f"Class 1: {len(class_1)} images")
print(f"Class 2: {len(class_2)} images")
print(f"Class 3: {len(class_3)} images")

batch_size = 32
img_height = 200
img_width = 113
NUM_EPOCHS = 10

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("List of class names:", class_names)

for image_batch, labels_batch in train_ds:
    print("Tensor shape for image batch:", image_batch.shape)
    print("Tensor shape for labels batch:", labels_batch.shape)
    break

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# Pixel values should now be in [0,1]
print("Testing normalization layer:", np.min(first_image), np.max(first_image))

assert np.min(first_image) >= 0
assert np.max(first_image) <= 1

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dropout(rate=0.5),
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS
)

model.save('v2_fingers_1.keras')