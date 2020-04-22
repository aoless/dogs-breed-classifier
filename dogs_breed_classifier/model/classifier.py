import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image

tf.get_logger().setLevel('INFO')
imagenet_resnet = tf.keras.applications.ResNet50(weights='imagenet')

def make_predictions(model, img_path, facecascade_path, dog_names):
    """
    Given model and image path predicts if there is human or dog on the image.
    Either way it predicts what breed of dog is it (or could be).

    Args:
        model - trained tensorflow model
        img_path - path to the image we want to predict
        facecascade_path - path to model for recognizing humans
        dog_names - list of all breed names

    Returns:
        String with prediction
    """

    answer = ""

    if _face_detector(img_path, facecascade_path):
        names, percentages = _predict_breed(img_path, model, dog_names)
        return f"I'm pretty sure that's human!/nBut as a dog it could be {names[0]}"
    elif _dog_detector(img_path, imagenet_resnet):
        names, percentages = _predict_breed(img_path, model, dog_names)
        return f"That's a dog! He looks like {names[0]}"
    else:
        return "I can't recognize what is that"


def _extract_resnet50(tensor):
    """Converts image tensor to resnet bottleneck feature"""
	return tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False
        ).predict(preprocess_input(tensor)
    )


def _predict_breed(img_path, model, dog_names):
    """Predics dog breed on image from image_path"""
    # extract bottleneck features
    bottleneck_feature = _extract_resnet50(_path_to_tensor(img_path))
    # obtain predicted vector
    model.summary()
    predicted_vector = model.predict(bottleneck_feature)[0]

    # return dog breed that is predicted by the model
    indices = predicted_vector.argsort()[-3:][::-1]
    names = [dog_names[index] for index in indices]
    percentages = [100 * predicted_vector[index] for index in indices]
    
    return names, percentages


def _face_detector(img_path, facecascade_path):
    """Detects human faces, returns true if face will be found at image"""
    facecascade = cv2.CascadeClassifier(facecascade_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray)

    return len(faces) > 0


def _dog_detector(img_path, imagenet_resnet):
    """Detects if there is dog at the image"""
    img = preprocess_input(_path_to_tensor(img_path))
    prediction = np.argmax(imagenet_resnet.predict(img))

    return ((prediction <= 268) & (prediction >= 151))


def _path_to_tensor(img_path):
    """Converts image path to image tensor"""
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


