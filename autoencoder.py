from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    LeakyReLU
)
from tensorflow.keras import regularizers


def build_autoencoder(n_inputs):

    visible = Input(shape=(n_inputs,))

    # Encoder
    x = Dense(
        n_inputs * 2,
        kernel_regularizer=regularizers.l1(0.01)
    )(visible)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(
        n_inputs,
        kernel_regularizer=regularizers.l1(0.01)
    )(x)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    bottleneck_size = round(n_inputs / 2)

    bottleneck = Dense(
        bottleneck_size,
        name="bottleneck"
    )(x)

    # Decoder
    x = Dense(
        n_inputs,
        kernel_regularizer=regularizers.l1(0.01)
    )(bottleneck)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(
        n_inputs * 2,
        kernel_regularizer=regularizers.l1(0.01)
    )(x)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    outputs = Dense(
        n_inputs,
        activation="linear"
    )(x)

    autoencoder = Model(
        visible,
        outputs,
        name="AutoEncoder"
    )

    encoder = Model(
        visible,
        bottleneck,
        name="Encoder"
    )

    return autoencoder, encoder



    from models.autoencoder import build_autoencoder

n_inputs = X.shape[1]

autoencoder, encoder = build_autoencoder(n_inputs)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

history = autoencoder.fit(
    x_train,
    x_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_test, x_test),
    verbose=2
)

encoder.save("encoder.h5")
