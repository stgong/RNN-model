# -*- coding: utf-8 -*-
from __future__ import print_function

from numpy.random import seed
seed(1)
import logging
logging.getLogger('tensorflow').disabled = True

import glob
import os
import random
import re
import sys
from time import time

import numpy as np
import pickle
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from .sequence_noise import SequenceNoise
from .target_selection import SelectTargets
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from helpers import evaluation


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
                        'sps_short': {'direction': 1},
                        'sps_long': {'direction': 1},
                        'user_coverage': {'direction': 1},
                        'item_coverage': {'direction': 1},
                        'ndcg': {'direction': 1},
                        'blockbuster_share': {'direction': -1},
                        'intra_list_similarity': {'direction': 1}
                        }



    def _common_filename(self, epochs):
        '''Common parts of the filename across sub classes.
		'''
        filename = "ml" + str(self.max_length) + "_bs" + str(self.batch_size) + "_ne" + str(
            epochs) + "_" + self.recurrent_layer.name + "_" + self.updater.name + "_" + self.target_selection.name

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

        # indices = tf.constant([[i[0]] for i in sequence], shape=(len(sequence),1))
        # updates = tf.constant(-np.inf, shape=(len(sequence),1))
        # output = tf.tensor_scatter_update(tf.reshape(output, (1477,1)), indices, updates)
        # output = tf.reshape(output, (1,1477))


        # output_array = output.numpy()
        # output_array[0][[i[0] for i in sequence]] = -np.inf
        # output = tf.convert_to_tensor(output_array)

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

    def train(self, dataset,tensorboard_callback,
              max_time=np.inf,
              progress=5000,
              autosave='Best',
              save_dir='',
              min_iterations=0,
              max_iter=4,
              load_last_model=False,
              early_stopping=None,
              validation_metrics=['sps']):

        time1 = time()
        self.dataset = dataset

        self.target_selection.set_dataset(dataset)

        if len(set(validation_metrics) & set(self.metrics.keys())) < len(validation_metrics):
            raise ValueError(
                'Incorrect validation metrics. Metrics must be chosen among: ' + ', '.join(self.metrics.keys()))

        # Load last model if needed
        iterations = 0
        epochs_offset = 0
        if load_last_model:
            epochs_offset = self.load_last(save_dir)

        X, Y = self._gen_mini_batch(self.dataset.dirname)
        x_val, y_val = self._gen_mini_batch(self.dataset.dirname, test=True)

        start_time = time()
        next_save = int(progress)
        # val_costs = []
        train_costs = []
        current_train_cost = []
        epochs = []
        metrics = {name: [] for name in self.metrics.keys()}
        filename = {}

        try:
            filepath = save_dir + self._get_model_filename(
                round(time() - time1, 3))

            checkpoint = ModelCheckpoint(filepath, verbose=1,
                                         monitor='val_loss', save_best_only=True, mode='auto')

            # history = self.model.fit_generator(batch_generator, epochs = min_iterations, steps_per_epoch= progress,
            #                                 validation_data = val_generator, validation_steps=20,
            #                                 # workers = 1, use_multiprocessing = True,
            #                                    callbacks= [checkpoint],
            #                                    verbose=2)

            history = self.model.fit(X,Y, epochs = min_iterations, batch_size = self.batch_size,
                                            validation_data = (x_val, y_val),
                                            # workers = 1, use_multiprocessing = True,
                                            #    callbacks= [checkpoint,tensorboard_callback],
                                               verbose=2)
            cost = history.history['loss']
            print(cost)

            current_train_cost = cost
            # print(current_train_cost)

            # Check if it is time to save the model
            epochs=[time()-start_time]

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
            self._print_progress(iterations, epochs[-1], start_time, train_costs
                                 , metrics, validation_metrics
                                 )
            # Save model
            run_nb = len(train_costs) - 1
            if autosave == 'All':
                filename[run_nb] = save_dir + self._get_model_filename(
                    round(epochs[-1], 3))
                self._save(filename[run_nb])
            elif autosave == 'Best':
                pareto_runs = self.get_pareto_front(metrics, validation_metrics)
                if run_nb in pareto_runs:
                    filename[run_nb] = save_dir + self._get_model_filename(
                        round(epochs[-1], 3))
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

    # def _gen_mini_batch(self, sequence_generator, test=False):
    #     ''' Takes a sequence generator and produce a mini batch generator.
    #     No subsequence, take full seq
    #     '''
    #     while True:
    #         j = 0
    #         sequences = []
    #         batch_size = self.batch_size
    #         if test:
    #             batch_size = 1
    #         while j < batch_size:  # j : user order
    #             sequence, user_id = next(sequence_generator)
    #             # print(user_id, len(sequence))
    #
    #             # finds the lengths of the different subsequences
    #             if len(sequence) <= self.max_length + 1:  # training set
    #                 sequence = sequence[0:-1]
    #                 target = self.target_selection(sequence[-1:], test=test)
    #                 sequences.append([user_id, sequence, target])
    #             else:
    #                 sequence = sequence[-self.max_length - 1:-1]
    #                 target = self.target_selection(sequence[-1:], test=test)
    #                 sequences.append([user_id, sequence, target])
    #
    #             j += 1
    #         yield self._prepare_input(sequences)

    def _gen_mini_batch(self, dirname, test=False):

        # while True:
        #     j = 0
        #     sequences = []
        #     batch_size = self.batch_size
        #     if test:
        #         batch_size = 1
        #     #     sequences = sequence_val_all
        #
        #     while j < batch_size:  # j : user order
        #     #
        #         if not test:
        #             sequence = next(self.batch_generator(dirname))
        #         else:
        #             sequence = next(self.batch_generator(dirname), test = True)
        #     #     if not test:
        #     #     sequences.append(sequence)
        #     #     else:
        #     #         sequences = sequence_val_all
        #     # else:
        #     #     sequences = sequence_train_all[j:j+batch_size]
        #     # sequences.append([user_id, sequence[start:start + l], target])
        #     # print([user_id, sequence[start:l], target])
        #     j += 1
        sequence_train_all = np.load(dirname + '/data/sub_sequences_all_list.pickle', allow_pickle=True)[0:32]
        sequence_val_all = np.load(dirname + '/data/validation_all_list.pickle', allow_pickle=True)[0:1]
        if not test:
            return self._prepare_input(sequence_train_all)
        else:
            return self._prepare_input(sequence_val_all)
            # print('mini_generator yielded a batch %d' % i)
            # i += 1

    # def batch_generator(self, dirname, test=False):
    #     for j in sequence_train_all:
    #         j = 0
    #         sequences = []
    #         batch_size = self.batch_size
    #
    #         if not test:
    #             sequences = sequence_train_all[j*batch_size: (j+1)*batch_size]
    #         else:
    #             sequences = sequence_val_all
    #         j += 1
    #         yield sequences

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

    def _get_model_filename(self, iterations):
        '''Return the name of the file to save the current model
		'''
        raise NotImplemented

    def prepare_networks(self):
        ''' Prepares the building blocks of the RNN, but does not compile them:
		self.l_in : input layer
		self.l_mask : mask of the input layer
		self.target : target of the network
		self.l_out : last output of the network
		self.cost : cost function

		and maybe others
		'''
        raise NotImplemented

    def _compile_train_network(self):
        ''' Compile self.train.
		self.train recieves a sequence and a target for every steps of the sequence, 
		compute error on every steps, update parameter and return global cost (i.e. the error).
		'''
        raise NotImplemented

    def _compile_predict_network(self):
        ''' Compile self.predict, the deterministic rnn that output the prediction at the end of the sequence
		'''
        raise NotImplemented

    def _save(self, filename):
        '''Save the parameters of a network into a file
		'''
        print('Save model in ' + filename)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

    def load_last(self, save_dir):
        '''Load last model from dir
		'''

        def extract_number_of_batches(filename):
            m = re.search('_nb([0-9]+)_', filename)
            return int(m.group(1))

        def extract_number_of_epochs(filename):
            m = re.search('_ne([0-9]+(\.[0-9]+)?)_', filename)
            return float(m.group(1))

        # Get all the models for this RNN
        file = save_dir + self._get_model_filename("*")
        file = np.array(glob.glob(file))

        if len(file) == 0:
            print('No previous model, starting from scratch')
            return 0

        # Find last model and load it
        last_batch = np.amax(np.array(map(extract_number_of_epochs, file)))
        last_model = save_dir + self._get_model_filename(last_batch)
        print('Starting from model ' + last_model)
        self.load(last_model)

        return last_batch

    def _load(self, filename):
        '''Load parameters values from a file
		'''

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
