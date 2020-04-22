from glob import glob

import numpy as np


def load_dataset(path, num_of_classes):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = tf.keras.utils.to_categorical(np.array(data['target']), num_of_classes)
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