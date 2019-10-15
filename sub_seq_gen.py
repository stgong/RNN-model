# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import os
import random
import re
import sys
from time import time

import numpy as np

from .sequence_noise import SequenceNoise
from .target_selection import SelectTargets
from keras.models import Sequential, load_model, Model
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
                        'user_coverage': {'direction': 1},
                        'item_coverage': {'direction': 1},
                        'ndcg': {'direction': 1},
                        'blockbuster_share': {'direction': -1}
                        }

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.target_selection.set_dataset(dataset)

    def _gen_mini_batch(self, sequence_generator, test=False):
        ''' Takes a sequence generator and produce a mini batch generator.
		The mini batch have a size defined by self.batch_size, and have format of the input layer of the rnn.
        with max_reuse_sequence = inf, one sequence will be used to make the whole batch (if the sequence is long enough)
        with max_reuse_sequence = 1, each sequence is used only once in the batch

		test determines how the sequence is splitted between training and testing
			test == False, the sequence is split randomly
			test == True, the sequence is split in the middle

		if test == False, max_reuse_sequence determines how many time a single sequence is used in the same batch.

		N.B. if test == True, max_reuse_sequence = 1 is used anyway
		'''
        i = 0
        uid = []
        sequences = []

        sequence, user_id = next(sequence_generator)
        uid.append(user_id)

        # finds the lengths of the different subsequences
        if not test:  # training set
            # seq_lengths = sorted(
            #     random.sample(range(2, len(sequence)),  # range
            #                   min([self.batch_size - j, len(sequence) - 2]))  # population
            # )
            seq_lengths = sorted(
                random.sample(range(10, len(sequence)), # range, min_sub_seq_length = 10
                              len(sequence) // 10))  # number of sub sequences generated for each user

            # print('called sequence generator', 'j =', j, 'user_id:', user_id,'seq_len=',len(sequence), 'seq_lengths =',len(seq_lengths))
        else:  # validating set
            seq_lengths = [int(len(sequence) / 2)]  # half of len

        start_l = []
        skipped_seq = 0
        for l in seq_lengths:
            # target is only for rnn with hinge, logit and logsig.
            target = sequence[l:][0]
            if len(target) == 0:
                skipped_seq += 1
                continue
            start = max(0, l - self.max_length)  # sequences cannot be longer than self.max_length
            start_l.append(start)
            # print(target)
            sequences.append([user_id, sequence[start:l], target])
        print(user_id, len(sequence), seq_lengths, start_l)

        skipped_seq = 0
        for l in seq_lengths:
            l = min(self.max_length, l)
            start = np.random.randint(0, len(sequence))  # randomly choose a start position
            start = min(start, len(sequence) - l)
            target = self.target_selection(sequence[start + l:], test=test)
            # target is only for rnn with hinge, logit and logsig.
            # target = self.target_selection(sequence[l:], test=test)
            if len(target) == 0:
                skipped_seq += 1
                continue
            # start = max(0, l - self.max_length)  # sequences cannot be longer than self.max_length
            # print(target)
            sequences.append([user_id, sequence[start:start + l], target])
            # sequences.append([user_id, sequence[start:l], target])
        # print([user_id, sequence[start:l], target])

        i += 1
        if test:
            yield self._prepare_input(sequences), [i[0] for i in sequence[seq_lengths[0]:]]
        else:
            return sequences

    # def _gen_mini_batch(self, sequence_generator, test=False):
    #     ''' Takes a sequence generator and produce a mini batch generator.
    #     The mini batch have a size defined by self.batch_size, and have format of the input layer of the rnn.
    #
    #     test determines how the sequence is splitted between training and testing
    #         test == False, the sequence is split randomly
    #         test == True, the sequence is split in the middle
    #
    #     if test == False, max_reuse_sequence determines how many time a single sequence is used in the same batch.
    #         with max_reuse_sequence = inf, one sequence will be used to make the whole batch (if the sequence is long enough)
    #         with max_reuse_sequence = 1, each sequence is used only once in the batch
    #     N.B. if test == True, max_reuse_sequence = 1 is used anyway
    #     '''
    #
    #     while True:
    #         j = 0
    #         sequences = []
    #         batch_size = self.batch_size
    #         if test:
    #             batch_size = 1
    #         while j < batch_size:  # j : user order
    #
    #             sequence, user_id = next(sequence_generator)
    #
    #             # finds the lengths of the different subsequences
    #             if not test:  # training set
    #                 seq_lengths = sorted(
    #                     random.sample(range(2, len(sequence)),  # range
    #                                   min([self.batch_size - j, len(sequence) - 2]))  # population
    #                 )
    #             elif self.iter:
    #                 batch_size = len(sequence) - 1
    #                 seq_lengths = list(range(1, len(sequence)))
    #             else:  # validating set
    #                 seq_lengths = [int(len(sequence) / 2)]  # half of len
    #
    #             skipped_seq = 0
    #             for l in seq_lengths:
    #                 # target is only for rnn with hinge, logit and logsig.
    #                 target = self.target_selection(sequence[l:], test=test)
    #                 if len(target) == 0:
    #                     skipped_seq += 1
    #                     continue
    #                 start = max(0, l - self.max_length)  # sequences cannot be longer than self.max_length
    #                 # print(target)
    #                 sequences.append([user_id, sequence[start:l], target])
    #             # print([user_id, sequence[start:l], target])
    #
    #             j += len(seq_lengths) - skipped_seq
    #
    #         if test:
    #             yield self._prepare_input(sequences), [i[0] for i in sequence[seq_lengths[0]:]]
    #         else:
    #             yield self._prepare_input(sequences)

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

    # Print on stderr for easier recording of progress
    # print(iterations, epochs, time() - start_time, train_costs[-1],
    # 	  ' '.join(map(str, [metrics[m][-1] for m in self.metrics])), file=sys.stderr)

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
