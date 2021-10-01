
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout


class ModelMnistClassifier(Sequential):
    def __init__(self):
        super().__init__([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            # Dropout(0.2),
            Dense(10)
        ])
