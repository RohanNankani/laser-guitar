import tensorflow as tf
models = tf.keras.models
layers = tf.keras.layers
# from tensorflow.keras.datasets import mnist
mnist = tf.keras.datasets.mnist

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Convert labels to binary (0 or 1)
train_labels = train_labels == 0  # True for zeros, False for ones
test_labels = test_labels == 0

# Build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

"""
PREDICTION STAGE
"""
# Assuming you have a new image in the variable 'new_image'
# Make sure to preprocess the new image similar to the training data
new_image = test_images[0].reshape((1, 28, 28, 1)).astype('float32') / 255

# Use the trained model to make predictions
prediction = model.predict(new_image)

# The prediction will be a probability, you can round it to get the predicted class (0 or 1)
predicted_class = round(prediction[0][0])

# Print the result
print("Predicted class:", predicted_class)
