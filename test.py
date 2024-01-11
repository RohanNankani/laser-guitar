# Load and preprocess the new image
import tensorflow as tf
image = tf.keras.preprocessing.image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('v2_fingers_1.keras')

# img_width, img_height = 113, 200

test_image_0_path = './finger_videos/training/class_3/image_1.png'
new_image = image.load_img(test_image_0_path)#, target_size=(img_width, img_height))

new_image_array = image.img_to_array(new_image)
new_image_array = np.expand_dims(new_image_array, axis=0)
new_image_array /= 255.

print(tf.shape(new_image_array))
# exit(0)


# prediction = model.predict(new_image_array)
# print(prediction)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(new_image_array)

print(predictions[0])


# ------------------
# predicted_class = (prediction[0][0])

# Print the result
# print("Predicted class:", predicted_class)
