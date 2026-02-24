"""
Model definitions for Pneumonia detection.
Includes a custom CNN and InceptionV3 transfer learning model.
"""

from keras.models import Model, Sequential
from keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten,
                          BatchNormalization, Dense)
from keras.applications.inception_v3 import InceptionV3


def get_conv_model():
    """
    Builds a custom deep CNN for binary classification (Normal vs. Pneumonia).

    Architecture:
        - 5 convolutional blocks (Conv2D + Conv2D + MaxPooling2D)
        - Flatten + Dense(64) + Dropout(0.4) + Dense(2, softmax)

    Returns:
        keras.models.Sequential: Compiled-ready sequential model.
    """
    model = Sequential()

    # Block 1
    model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(3, 150, 150)))
    model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(3, 150, 150)))
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 4
    model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 5
    model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Classifier head
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax'))

    print(model.summary())
    return model


def get_inception_model():
    """
    Builds an InceptionV3 transfer learning model for binary classification.

    Uses ImageNet pretrained weights with frozen base layers and a new
    BatchNormalization + Dense(2, softmax) head.

    Returns:
        keras.models.Model: Compiled-ready functional model.
    """
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

    x = base_model.output
    x = BatchNormalization()(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze all base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model.summary()
    return model
