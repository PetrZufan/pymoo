
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout


class ModelDibcoClassifier(Sequential):
    def __init__(self, img_size=25):
        # hidden_size = (img_size * img_size) / 8
        hidden_size = img_size
        super().__init__([
            Flatten(input_shape=(img_size, img_size)),
            Dense(hidden_size, activation='relu'),
            Dense(2)
        ])
