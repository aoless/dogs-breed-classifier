from glob import glob

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

def load_dataset(path, num_of_classes):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = to_categorical(np.array(data['target']), num_of_classes)
    return dog_files, dog_targets


def clean_label(label):
    return ''.join(filter(str.isalpha, label))


def extract_images_and_labels(data_dir):
    images, targets = load_dataset(data_dir)
    return (images, targets)


def create_labels_names(data_dir):
    dog_names = [
        clean_label(item.split('/')[-2]) for item in sorted(glob(f"{data_dir}/*/"))
        ]
    
    return dog_names


def load_pretrainted_resnet_features(path):
    """Loads and returns precomputed features for ResNet50 architecture"""
    bottleneck_features = np.load(path)

    train_resnet = bottleneck_features['train']
    valid_resnet = bottleneck_features['valid']
    test_resnet = bottleneck_features['test']

    return (train_resnet, valid_resnet, test_resnet)


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)