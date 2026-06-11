# model

import tensorflow as tf
from tensorflow.keras import layers, models


def ICUNet(input_shape):

    input_layer = layers.Input(shape=input_shape)

    # Initial Conv1D
    x = layers.Conv1D(
        filters=128,
        kernel_size=5,
        padding='same',
        activation='relu'
    )(input_layer)

    # Residual Block
    def residual_block(input_tensor):

        x = layers.Conv1D(
            128,
            5,
            padding='same',
            activation='relu'
        )(input_tensor)

        x = layers.Conv1D(
            128,
            5,
            padding='same'
        )(x)

        x = layers.Add()([x, input_tensor])

        return layers.Activation('relu')(x)

    for _ in range(4):
        x = residual_block(x)

    # Squeeze-and-Excitation
    def se_block(input_tensor):

        channels = input_tensor.shape[-1]

        se = layers.GlobalAveragePooling1D()(input_tensor)

        se = layers.Dense(
            channels // 16,
            activation='relu'
        )(se)

        se = layers.Dense(
            channels,
            activation='sigmoid'
        )(se)

        se = layers.Reshape((1, channels))(se)

        return layers.Multiply()([input_tensor, se])

    x = se_block(x)

    # Self-Attention
    def self_attention(input_tensor):

        query = layers.Dense(input_tensor.shape[-1])(input_tensor)
        key = layers.Dense(input_tensor.shape[-1])(input_tensor)
        value = layers.Dense(input_tensor.shape[-1])(input_tensor)

        scores = layers.Dot(axes=-1)([query, key])

        scores = scores / tf.sqrt(
            tf.cast(input_tensor.shape[-1], tf.float32)
        )

        weights = layers.Activation('softmax')(scores)

        attention = layers.Dot(
            axes=[-1, 1]
        )([weights, value])

        return layers.Dense(
            input_tensor.shape[-1]
        )(attention)

    x = self_attention(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Dense(
        128,
        activation='relu'
    )(x)

    output_layer = layers.Dense(
        1,
        activation='sigmoid'
    )(x)

    return models.Model(
        inputs=input_layer,
        outputs=output_layer
    )
