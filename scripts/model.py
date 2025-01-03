from tensorflow.keras import layers, models

def build_model(input_shape=(512, 512, 3)):
    """
    Build a simple CNN-based model for edge detection.
    """
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    return models.Model(inputs, outputs)
