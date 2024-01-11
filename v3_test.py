import tensorflow as tf
image = tf.keras.preprocessing.image
import numpy as np
import PIL

def predict(new_image_path: str, model_path: str):
    model = tf.keras.models.load_model(model_path)

    class_names = ['class_1', 'class_2', 'class_3']

    new_image_path = new_image_path
    PIL.Image.open(new_image_path)

    image = tf.keras.preprocessing.image
    new_image = image.load_img(new_image_path)

    new_image_array = image.img_to_array(new_image)
    new_image_array = np.expand_dims(new_image_array, axis=0)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(new_image_array)

    print("Prediction array:", predictions[0])
    idx = np.argmax(predictions[0])
    print("Highest index:", idx)
    print("Predicted label:", class_names[idx])
