import numpy as np

from keras.utils import to_categorical


class KerasHelper(object):

    def __init__(self, data: [int], number_of_steps: int, batch_size: int, vocabulary_size: int, skip_step: int):
        self.data = data
        self.number_of_steps = number_of_steps
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.current_index = 0
        self.skip_step = skip_step

    def generate(self) -> (np.ndarray, np.ndarray):
        x = np.zeros((self.batch_size, self.number_of_steps))
        y = np.zeros((self.batch_size, self.number_of_steps, self.vocabulary_size))
        while True:
            for i in range(self.batch_size):
                if self.current_index + self.number_of_steps >= len(self.data):
                    self.current_index = 0
                x[i, :] = self.data[self.current_index:self.current_index + self.number_of_steps]
                temp = self.data[self.current_index + 1:self.current_index + self.number_of_steps + 1]
                y[i, :, :] = to_categorical(temp, num_classes=self.vocabulary_size)
                self.current_index += self.skip_step
            yield x, y
