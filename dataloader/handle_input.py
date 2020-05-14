import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from utils.logger import Logger


class InputHandler(Logger):
    def __init__(self, input_path, split_vali=False, batch_size=256, test_size=0.3):
        self.input_path = input_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.initial_data_shape = None
        self.split_vali = split_vali

    def create_tf_dataset(self):
        x_train, x_test, x_val = self.split_train_val_test(self.read_data())
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        test_dataset = tf.data.Dataset.from_tensor_slices(x_test)

        train_dataset = train_dataset.shuffle(x_train.shape[0])
        train_dataset = train_dataset.batch(self.batch_size)

        test_dataset = test_dataset.batch(self.batch_size)

        if self.split_vali:
            val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
            val_dataset = val_dataset.batch(self.batch_size)
            return train_dataset, test_dataset, val_dataset
        return train_dataset, test_dataset, None

    def split_train_val_test(self, data):
        x_train, x_test, _, _ = train_test_split(data, data, test_size=self.test_size)
        if self.split_vali:
            x_test, x_val, _, _ = train_test_split(x_test, x_test, test_size=0.5)
            return x_train, x_test, x_val
        return x_train, x_test, None

    def read_data(self):
        inp = np.load(self.input_path)
        inp = inp.astype(np.float32)
        # length = int(inp.shape[0] / 1000)
        # inp = inp[:length]
        self.initial_data_shape = inp.shape
        return inp




