import tensorflow as tf
image = tf.keras.preprocessing.image
import numpy as np
import PIL

def predict():
    model = tf.keras.models.load_model('v3_fingers_2_mac.keras')

    class_names = ['class_1', 'class_2', 'class_3']

    new_image_path = "./finger_videos/testing/3-remove-bg.png"
    # new_image_path = "./finger_videos/training/class_2/image_355.png"
    PIL.Image.open(new_image_path)

    image = tf.keras.preprocessing.image
    new_image = image.load_img(new_image_path)

    print(new_image)

    # print("New image:", plt.imread(new_image_path))

    new_image_array = image.img_to_array(new_image)
    new_image_array = np.expand_dims(new_image_array, axis=0)
    # new_image_array /= 255.

    # print(tf.shape(new_image_array))
    # print(new_image_array)

    # prediction = model.predict(new_image_array)
    # print("Prediction:", prediction)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(new_image_array)

    print("Prediction array:", predictions[0])
    idx = np.argmax(predictions[0])
    print("Highest index:", idx)
    print("Predicted label:", class_names[idx])