import numpy as np
from tensorflow import keras
from keras.utils import Sequence

# from __future__ import division
import numpy as np


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.list = list
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_temp = [self.list[k] for k in indexes]

        # Generate data
        sequences = self.__data_generation(list_temp)

        return sequences

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list))

    def _prepare_input(self, sequences):
        """ Sequences is a list of [user_id, input_sequence, targets]
        """
        # print("_prepare_input()")
        batch_size = len(sequences)

        # Shape of return variables
        if self.recurrent_layer.embedding_size > 0:
            X = np.zeros((batch_size, self.max_length),
                         dtype=self._input_type)  # keras embedding requires movie-id sequence, not one-hot
        else:
            X = np.zeros((batch_size, self.max_length, self.n_items), dtype=self._input_type)  # input of the RNN
        Y = np.zeros((batch_size, self.n_items), dtype='float32')  # output target

        for i, sequence in enumerate(sequences):
            user_id, in_seq, target = sequence

            if self.recurrent_layer.embedding_size > 0:
                X[i, :len(in_seq)] = np.array([item[0] for item in in_seq])
            else:
                seq_features = np.array(list(map(lambda x: self._get_features(x), in_seq)))
                X[i, :len(in_seq), :] = seq_features  # Copy sequences into X

            # Becareful, target is a list of multiple target tuples before
            # Using preprocessed sub-sequences, need to user Y[i][target[0]] = 1.
            # Y[i][target[0][0]] = 1.
            Y[i][target[0]] = 1.

        return X, Y

    def __data_generation(self, list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        sequences = []

        # Generate data
        for subseq in list_temp:
            sequences.append(subseq)
        return self._prepare_input(sequences)