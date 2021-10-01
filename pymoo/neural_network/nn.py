
import tensorflow as tf
import numpy as np


class NN:
    def __init__(self):
        pass

    def load_dataset(self, dataset=tf.keras.datasets.mnist):
        # dataset = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        return (train_images, train_labels), (test_images, test_labels)

    def get_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])

        return model

    def run(self):
        # load dataset
        (train_images, train_labels), (test_images, test_labels) = self.load_dataset()

        # define model
        model = self.get_model()

        # define other metrics
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      # loss=tf.losses.MeanSquaredError(),
                      # loss=tf.keras.losses.MeanAbsoluteError(),
                      metrics=['accuracy'])

        # train model on train-dataset
        model.fit(train_images, train_labels, epochs=10)

        # evaluate trained model on test-dataset
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        return

    def single(self):
        # load dataset
        (train_images, train_labels), (test_images, test_labels) = self.load_dataset()

        # define model
        model = self.get_model()

        # predict one sample
        prediction = model(train_images[:1]).numpy()

        # define loss function
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # calculate loss
        loss = loss_fn(train_labels[:1], prediction).numpy()
        return


if __name__ == "__main__":
    print(tf.__version__)

    nn = NN()
    nn.single()
