import tensorflow as tf


def __contracting_step(filters, inputs, is_bottom=False):
    c = tf.keras.layers.Conv2D(
        filters,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(inputs)
    c = tf.keras.layers.Dropout(0.1)(c)
    c = tf.keras.layers.Conv2D(
        filters,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c)
    if is_bottom:
        return [c, None]
    p = tf.keras.layers.MaxPooling2D((2, 2))(c)

    return [c, p]


def __expansive_step(filters, inputs_c, constracting_c):
    u = tf.keras.layers.Conv2DTranspose(
        filters, (2, 2), strides=(2, 2), padding="same"
    )(inputs_c)
    u = tf.keras.layers.concatenate([u, constracting_c])
    c = tf.keras.layers.Conv2D(
        filters,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u)
    c = tf.keras.layers.Dropout(0.2)(c)
    c = tf.keras.layers.Conv2D(
        filters,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c)

    return c


def build_model(input_shape):
    # Build the model
    inputs = tf.keras.layers.Input(input_shape)

    filters = 16
    # Contraction path
    c1, p1 = __contracting_step(filters, inputs)
    filters *= 2
    c2, p2 = __contracting_step(filters, p1)
    filters *= 2
    c3, p3 = __contracting_step(filters, p2)
    filters *= 2
    c4, p4 = __contracting_step(filters, p3)
    filters *= 2
    c5, _ = __contracting_step(filters, p4, is_bottom=True)

    # Expansive path
    filters /= 2
    c6 = __expansive_step(filters, c5, c4)
    filters /= 2
    c7 = __expansive_step(filters, c6, c3)
    filters /= 2
    c8 = __expansive_step(filters, c7, c2)
    filters /= 2
    c9 = __expansive_step(filters, c8, c1)

    outputs = tf.keras.layers.Conv2D(3, (1, 1), activation="sigmoid")(c9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    model = build_model((256, 256, 3))
    model.summary()
