import tensorflow as tf

def load_pretrained_model(num_of_outputs, checkpoint_path):
    resnet_model = create(num_of_outputs)
    resnet_model.load_weights(checkpoint_path)

    return resnet_model


def create(num_of_outputs):
    """
    Creates custom ResNet50 architecture using transfer learning.

    Args:
        num_of_outpus - number of classes which modified model should classify

    Returns:
        ResNet50 model with additional layers at the end
    """
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(
        input_shape=(7, 7, 2048),
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

    resnet_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
        )

    return resnet_model
