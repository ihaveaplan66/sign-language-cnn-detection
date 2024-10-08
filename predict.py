import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(img_path, model, train):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_names = list(train.class_indices.keys())
    return class_names[class_idx]
