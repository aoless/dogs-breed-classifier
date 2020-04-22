import tensorflow as tf

def create(bottleneck_shape, num_of_outputs):
    """
    Creates custom ResNet50 architecture using transfer learning.

    Args:
        bottleneck_shape - size of embeddings from orginal ResNet50 model
        num_of_outpus - number of classes which modified model should classify

    Returns:
        ResNet50 model with additional layers at the end
    """
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(
        input_shape=bottleneck_shape,
        )
    dense = tf.keras.layers.Dense(512)
    dropout = tf.keras.layers.Dropout(0.4)
    prediction_layer = tf.keras.layers.Dense(num_of_outputs, activation="softmax")

    resnet_model = tf.keras.Sequential([
        global_average_layer,
        dense,
        dropout,
        prediction_layer,
    ])

    return resnet_model
    

def load_weights(model, path):
    try:
        model.load_weights(path)
    else:
        print("""Couldn't load weights into a provided model. 
        Wrong architecture or file path""")

    return model

