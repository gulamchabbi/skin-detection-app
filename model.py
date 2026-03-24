import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("skin_model.h5")

class_names = ["Acne", "Normal"]

def predict_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0]

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[predicted_index]) * 100

    return predicted_class, round(confidence, 2)