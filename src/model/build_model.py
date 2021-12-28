import tensorflow as tf
import keras_efficientnet_v2


def build_model(initial_bias=None):
    output_bias = None
    if initial_bias is not None:
        output_bias = tf.keras.initializers.Constant(initial_bias)

    # Model sizes: S, M, L, XL
    base_model = keras_efficientnet_v2.EfficientNetV2M(
        include_preprocessing=True,
        input_shape=(None, None, 3),
        pretrained="imagenet21k-ft1k",
        num_classes=0,  # remove top layer
    )
    base_model.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid',
                              bias_initializer=output_bias)
    ])

    return model
