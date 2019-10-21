import collections
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Activation, Dense, Dropout, Embedding, GRU, TimeDistributed
from keras.models import load_model, Sequential
from scipy import signal

from helper_config import ConfigHelper
from helper_convert import ConvertHelper
from helper_keras import KerasHelper


class Bach:
    _cfg = ConfigHelper().config[0]

    def __init__(self, option: str = 'train'):
        ConvertHelper.generate_input()

        training_data, validation_data, vocabulary_size, map_direct, map_reverse = Bach._load_data()
        training_data_generator = KerasHelper(training_data,
                                              Bach._cfg['number_of_steps'],
                                              Bach._cfg['batch_size'],
                                              vocabulary_size,
                                              skip_step=Bach._cfg['number_of_steps'])
        validation_data_generator = KerasHelper(validation_data,
                                                Bach._cfg['number_of_steps'],
                                                Bach._cfg['batch_size'],
                                                vocabulary_size,
                                                skip_step=Bach._cfg['number_of_steps'])

        model = Sequential()
        model.add(Embedding(vocabulary_size,
                            Bach._cfg['hidden_size'],
                            input_length=Bach._cfg['number_of_steps']))
        model.add(GRU(Bach._cfg['hidden_size'], return_sequences=True))
        model.add(GRU(Bach._cfg['hidden_size'], return_sequences=True))
        model.add(GRU(Bach._cfg['hidden_size'], return_sequences=True))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(vocabulary_size)))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

        checkpoint = ModelCheckpoint(filepath=Bach._cfg['data_path'] + '/model_epoch_{epoch:03d}.hdf5', verbose=1)
        logger = CSVLogger(Bach._cfg['data_path'] + '/log.csv', append=False, separator=',')

        if option == 'train':
            model.fit_generator(training_data_generator.generate(),
                                len(training_data) // (Bach._cfg['batch_size'] * Bach._cfg['number_of_steps']),
                                Bach._cfg['number_of_epochs'],
                                validation_data=validation_data_generator.generate(),
                                validation_steps=len(validation_data) // (
                                        Bach._cfg['batch_size'] * Bach._cfg['number_of_steps']),
                                callbacks=[checkpoint, logger])
            model.save(Bach._cfg['data_path'] + '/model_final.hdf5')

        if option == 'test':
            model = load_model(Bach._cfg['data_path'] + '/model_final.hdf5')

            with open('inception.txt', 'r') as f:
                inception = f.read()

            random.seed(Bach._cfg['seed'])
            inception = inception.replace('\n', ' <eos> ')
            inception_tokens = inception.split()
            sentence_ids = [map_direct[element] for element in inception_tokens if element in map_direct]
            sentence = inception
            for n in range(Bach._cfg['number_of_predictions']):
                i = np.zeros((1, Bach._cfg['number_of_steps']))
                i[0] = np.array(sentence_ids[-Bach._cfg['number_of_steps']:])
                prediction = model.predict(i)
                o = np.argsort(prediction[:, Bach._cfg['number_of_steps'] - 1, :]).flatten()[::-1]

                rnd = random.random()
                idx = 0
                temperature = Bach._cfg['temperature']
                while rnd < 1 / temperature and idx < vocabulary_size:
                    idx += 1
                    temperature += 1
                if idx == vocabulary_size:
                    idx = 0
                w = o[idx]

                sentence_ids.append(w)
                sentence += map_reverse[w]
                sentence += ' '

            with open(Bach._cfg['data_path'] + '/output.nwctxt', 'w') as f:
                f.write(sentence)
            ConvertHelper.generate_output()

        if option == 'fft':
            Bach._plot_fft(training_data + validation_data)

    @staticmethod
    def _read_words(filename: str) -> [str]:
        with tf.io.gfile.GFile(filename, 'r') as f:
            return f.read().replace('\n', '<eos>').split()

    @staticmethod
    def _build_vocabulary(filename: str) -> dict:
        data = Bach._read_words(filename)
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        elements, _ = list(zip(*count_pairs))
        map_direct = dict(zip(elements, range(len(elements))))
        return map_direct

    @staticmethod
    def _file_to_ids(filename: str, map_direct: dict) -> [int]:
        data = Bach._read_words(filename)
        return [map_direct[element] for element in data if element in map_direct]

    @staticmethod
    def _load_data() -> ([int], [int], int, dict, dict):
        training_path = os.path.join(Bach._cfg['data_path'], 'training.txt')
        validation_path = os.path.join(Bach._cfg['data_path'], 'validation.txt')
        map_direct = Bach._build_vocabulary(training_path)
        training_data = Bach._file_to_ids(training_path, map_direct)
        validation_data = Bach._file_to_ids(validation_path, map_direct)
        vocabulary_size = len(map_direct)
        map_reverse = dict(zip(map_direct.values(), map_direct.keys()))
        return training_data, validation_data, vocabulary_size, map_direct, map_reverse

    @staticmethod
    def _plot_fft(all_data: [int], smooth: bool = True):
        sample_points = len(all_data)
        xf = np.linspace(0, sample_points, sample_points)
        sig = np.abs(np.fft.fft(all_data))
        sig[0] = 0
        sig[-1] = 0
        if smooth:
            sig = signal.wiener(sig)
        yf = sig / np.max(sig)
        plt.plot(xf[:sample_points // 2], yf[:sample_points // 2])
        plt.grid()
        plt.show()


Bach('test')
