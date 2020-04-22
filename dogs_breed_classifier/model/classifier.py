from dogs_breed_classifier.data import data_loader


imagenet_resnet = tf.keras.applications.ResNet50(weights='imagenet')

def make_predictions(model, img_path, facecascade_path):
    if _face_detector(img_path, facecascade_path):
        names, percentages = _predict_breed(img_path)
        answer = ""
        for name, pct in zip(names, percentages):
            answer += f"\n{name} ({pct:.4} %)"
        return f"I'm pretty sure that's human!\nBut as a dog it could be... {answer}"
    elif dog_detector(img_path):
        names, percentages = _predict_breed(img_path)
        answer = ""
        for name, pct in zip(names, percentages):
            answer += f"\n{name} ({pct:.4} %)"
        return f"That's a dog! and he looks like... {answer}"
    else:
        return "I can't recognize what is that"


def _extract_resnet50(tensor):
	return tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False
        ).predict(preprocess_input(tensor)
    )


def _predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = _extract_resnet50(data_loader.path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = resnet_model.predict(bottleneck_feature)[0]

    # return dog breed that is predicted by the model
    indices = predicted_vector.argsort()[-3:][::-1]
    names = [dog_names[index] for index in indices]
    percentages = [100 * predicted_vector[index] for index in indices]
    
    return names, percentages


def _face_detector(img_path, facecascade_path):
    facecascade = cv2.CascadeClassifier(facecascade_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray)

    return len(faces) > 0


def _dog_detector(img_path, imagenet_resnet):
    img = preprocess_input(data_loader.path_to_tensor(img_path))
    prediction = np.argmax(imagenet_resnet.predict(img))

    return ((prediction <= 268) & (prediction >= 151))



