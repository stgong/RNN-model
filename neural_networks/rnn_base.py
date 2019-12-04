# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import os
import random
import re
import sys
from time import time

import numpy as np
import pickle


from keras.utils import Sequence
from .sequence_noise import SequenceNoise
from .target_selection import SelectTargets
from keras.callbacks import ModelCheckpoint

# from helpers.data_generator import DataGenerator


class RNNBase(object):
    """Base for RNN object.
	"""

    def __init__(self,
                 sequence_noise=SequenceNoise(),
                 target_selection=SelectTargets(),
                 active_f='tanh',
                 max_length=30,
                 batch_size=16,
                 tying=False,
                 temperature=10,
                 gamma=0.5,
                 iter=False,
                 tying_new=False,
                 attention=False):

        super(RNNBase, self).__init__()

        self.max_length = max_length
        self.batch_size = batch_size
        self.sequence_noise = sequence_noise
        self.target_selection = target_selection
        self.active_f = active_f
        self.tying = tying
        self.temperature = temperature
        self.gamma = gamma
        self.iter = iter
        self.tying_new = tying_new
        self.attention = attention

        self._input_type = 'float32'

        self.name = "RNN base"
        self.metrics = {'recall': {'direction': 1},
                        'precision': {'direction': 1},
                        'sps': {'direction': 1},
                        'user_coverage': {'direction': 1},
                        'item_coverage': {'direction': 1},
                        'ndcg': {'direction': 1},
                        'blockbuster_share': {'direction': -1}
                        }



    def _common_filename(self, version, true_epochs):
        '''Common parts of the filename across sub classes.
		'''
        filename = "ml" + str(self.max_length) + "_bs" + str(self.batch_size) + self.recurrent_layer.name + "_" + self.updater.name + "_" + self.target_selection.name + "_epoch"\
                   + str(true_epochs) + "_i" + str(version)

        if self.sequence_noise.name != "":
            filename += "_" + self.sequence_noise.name

        if self.active_f != 'tanh':
            filename += "_act" + self.active_f[0].upper()
        if self.tying:
            filename += "_ty_tp" + str(self.temperature) + "_gm" + str(self.gamma)
            if self.tying_new:
                filename += "_new"
        if self.iter:
            filename += "_it"
        if self.attention:
            filename += "_att"
        return filename

    def top_k_recommendations(self, sequence, k=10, exclude=None):
        ''' Receives a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''

        seq_by_max_length = sequence[-min(self.max_length, len(sequence)):]  # last max length or all

        # Prepare RNN input
        if self.recurrent_layer.embedding_size > 0:
            X = np.zeros((1, self.max_length), dtype=np.int32)  # ktf embedding requires movie-id sequence, not one-hot
            X[0, :len(seq_by_max_length)] = np.array([item[0] for item in seq_by_max_length])
        else:
            X = np.zeros((1, self.max_length, self._input_size()), dtype=self._input_type)  # input shape of the RNN
            X[0, :len(seq_by_max_length), :] = np.array(list(map(lambda x: self._get_features(x), seq_by_max_length)))

        # Run RNN

        output = self.model.predict_on_batch(X)

        # filter out viewed items
        output[0][[i[0] for i in sequence]] = -np.inf

        # exclude items given in args
        # output[exclude] = -np.inf

        # find top k according to output
        return list(np.argpartition(-output[0], list(range(k)))[:k])

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.target_selection.set_dataset(dataset)

    def get_pareto_front(self, metrics, metrics_names):
        costs = np.zeros((len(metrics[metrics_names[0]]), len(metrics_names)))
        for i, m in enumerate(metrics_names):
            costs[:, i] = np.array(metrics[m]) * self.metrics[m]['direction']
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)
        return np.where(is_efficient)[0].tolist()

    def train(self, dataset,
              max_time=np.inf,
              number_of_batches=10,
              autosave='Best',
              save_dir='',
              epochs=10,
              max_iter=4,
              early_stopping=None,
              validation_metrics=['sps'],
              debug = False):

        start_time = time()
        self.dataset = dataset
        batch_size = self.batch_size


        self.target_selection.set_dataset(dataset)

        if len(set(validation_metrics) & set(self.metrics.keys())) < len(validation_metrics):
            raise ValueError(
                'Incorrect validation metrics. Metrics must be chosen among: ' + ', '.join(self.metrics.keys()))

        train_subseq_list = np.load(self.dataset.dirname + '/data/train_subseq_list.pickle', allow_pickle=True)
        val_subseq_list = np.load(self.dataset.dirname + '/data/validation_subseq_list.pickle', allow_pickle=True)

        iterations = 0
        # val_costs = []
        train_costs = []
        current_train_cost = []
        metrics = {name: [] for name in self.metrics.keys()}
        filename = {}
        if debug:
            number_of_batches_input = number_of_batches
        else:
            number_of_batches_input = len(train_subseq_list) // self.batch_size
        try:

            ne = '{epoch:03d}'
            filepath = save_dir + self._get_model_filename(
                round(time() - start_time, 3),'{epoch:03d}')

            checkpoint = ModelCheckpoint(filepath,
                                         verbose=1,
                                         monitor='val_loss', save_best_only=True, mode='auto')


            batch_generator = DataGenerator(train_subseq_list, self.recurrent_layer.embedding_size, self.max_length, self._input_size())
            val_generator = DataGenerator(val_subseq_list, self.recurrent_layer.embedding_size, self.max_length, self._input_size())


            history = self.model.fit_generator(batch_generator,
                                            # steps_per_epoch=number_of_batches_input,
                                            validation_data=val_generator,
                                            # validation_steps=1,
                                            # validation_steps=len(val_subseq_list)//batch_size,
                                            workers = 4, use_multiprocessing = True,
                                               callbacks= [checkpoint],
                                               verbose=1)
            cost = history.history['loss']
            # print(cost)
            current_train_cost = cost
            # print(current_train_cost)

            # Check if it is time to save the model
            version=[time()-start_time]

            # Average train cost
            train_costs.append(np.mean(current_train_cost))
            current_train_cost = []

            # intermediate_model = Model(inputs=self.model.layers[0].input,
            # 						   outputs=[l.output for l in self.model.layers[1:]])

            # intermediate_output = intermediate_model.predict(batch[0])
            # print(intermediate_output)

            # Compute validation cost
            metrics = self._compute_validation_metrics(self.dataset, metrics)

            # Print info
            self._print_progress(number_of_batches_input, epochs, start_time, train_costs
                                 , metrics, validation_metrics
                                 )
            # Save model
            run_nb = len(train_costs) - 1
            if autosave == 'All':
                filename[run_nb] = save_dir + self._get_model_filename(
                    round(version[-1], 3), epochs)
                self._save(filename[run_nb])
            elif autosave == 'Best':
                pareto_runs = self.get_pareto_front(metrics, validation_metrics)
                if run_nb in pareto_runs:
                    filename[run_nb] = save_dir + self._get_model_filename(
                        round(version[-1], 3), epochs)
                    self._save(filename[run_nb])
                    to_delete = [r for r in filename if r not in pareto_runs]
                    for run in to_delete:
                        try:
                            os.remove(filename[run])
                        except OSError:
                            print('Warning : Previous model could not be deleted')
                        del filename[run]

        except KeyboardInterrupt:
            print('Training interrupted')

        best_run = np.argmax(
            np.array(metrics[validation_metrics[0]]) * self.metrics[validation_metrics[0]]['direction'])
        return ({m: metrics[m][best_run] for m in self.metrics.keys()}, time() - start_time, filename[best_run])


    def generator(self, dataset):

        samples_per_epoch = len(dataset)
        number_of_batches = samples_per_epoch// self.batch_size
        counter = 0
        batch_size = self.batch_size
        while True:

            sequences = dataset[batch_size * counter:batch_size * (counter + 1)]
            # print(counter)
            counter += 1
            yield self._prepare_input(sequences)

            # restart counter to yeild data in the next epoch as well
            if counter >= number_of_batches:
                counter = 0


    def _print_progress(self, iterations, epochs, start_time, train_costs
                        , metrics
                        , validation_metrics
                        ):
        '''Print learning progress in terminal
		'''
        print(self.name, iterations, "batchs, ", epochs, " epochs in", time() - start_time, "s")
        print("Last train cost : ", train_costs[-1])
        for m in self.metrics:
            print(m, ': ', metrics[m][-1])
            if m in validation_metrics:
                print('Best ', m, ': ',
                      max(np.array(metrics[m]) * self.metrics[m]['direction']) * self.metrics[m]['direction'])

        print('-----------------')

    def _save(self, filename):
        '''Save the parameters of a network into a file
		'''
        print('Save model in ' + filename)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))


    def _input_size(self):
        ''' Returns the number of input neurons
		'''
        return self.n_items

    def _get_features(self, item):
        '''Change a tuple (item_id, rating) into a list of features to feed into the RNN
		features have the following structure: [one_hot_encoding, personal_rating on a scale of ten, average_rating on a scale of ten, popularity on a log scale of ten]
		'''

        one_hot_encoding = np.zeros(self.n_items)
        one_hot_encoding[item[0]] = 1
        return one_hot_encoding

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list, emb, max_length, n_items, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.list = list
        self.on_epoch_end()
        self.emb = emb
        self.max_length = max_length
        self.n_items = n_items
        self._input_type = 'float32'

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # print(indexes)
        # Find list of IDs
        list_temp = [self.list[k] for k in indexes]

        # Generate data
        sequences = self.__data_generation(list_temp)

        return sequences

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list))

    def _get_features(self, item):
        '''Change a tuple (item_id, rating) into a list of features to feed into the RNN
        features have the following structure: [one_hot_encoding, personal_rating on a scale of ten, average_rating on a scale of ten, popularity on a log scale of ten]
        '''

        one_hot_encoding = np.zeros(self.n_items)
        one_hot_encoding[item[0]] = 1
        return one_hot_encoding
    def _prepare_input(self, sequences):
        """ Sequences is a list of [user_id, input_sequence, targets]
        """
        # print("_prepare_input()")
        batch_size = len(sequences)

        # Shape of return variables
        if self.emb > 0:
            X = np.zeros((batch_size, self.max_length),
                         dtype=self._input_type)  # keras embedding requires movie-id sequence, not one-hot
        else:
            X = np.zeros((batch_size, self.max_length, self.n_items), dtype=self._input_type)  # input of the RNN
        Y = np.zeros((batch_size, self.n_items), dtype='float32')  # output target

        for i, sequence in enumerate(sequences):
            user_id, in_seq, target = sequence

            if self.emb > 0:
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
