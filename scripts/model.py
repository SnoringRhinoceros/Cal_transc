from tensorflow.keras import layers, models

def build_model(input_shape=(512, 512, 1)):  # Adjusted for grayscale images
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)  # Downsamples to 256x256
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Downsamples to 128x128
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)  # Upsamples to 256x256
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)  # Upsamples to 512x512
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)  # Single-channel output
    return models.Model(inputs, outputs)
