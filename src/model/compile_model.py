import tensorflow as tf
import tensorflow_addons as tfa


def compile_model(model, num_classes, learning_rate=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(),
            tfa.metrics.F1Score(num_classes, average='macro'),
        ],
    )

    return model
