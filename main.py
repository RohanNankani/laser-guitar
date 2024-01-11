import tensorflow as tf
models = tf.keras.models
layers = tf.keras.layers
import numpy as np

# from tensorflow.keras.datasets import mnist
mnist = tf.keras.datasets.mnist

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
image = tf.keras.preprocessing.image

# Define the path to your dataset
train_data_dir = './videos/training'
img_width, img_height = 852, 480
batch_size = 25
epochs = 5

# Use the ImageDataGenerator to load and augment your images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# print(train_datagen)
# exit(0)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'  # Assuming a binary classification task
)

# Build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=epochs)

# Save the trained model if needed
model.save('model_v1.h5')

# Load and preprocess the new image
test_image_0_path = './videos/testing/0-2.png'
new_image = image.load_img(test_image_0_path, target_size=(img_width, img_height))

new_image_array = image.img_to_array(new_image)
new_image_array = np.expand_dims(new_image_array, axis=0)
new_image_array /= 255.

# Make predictions
prediction = model.predict(new_image_array)

# Interpret the prediction (assuming binary classification)
predicted_class = round(prediction[0][0])

# Print the result
print("Predicted class:", predicted_class)