from __future__ import print_function

from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

from helpers import evaluation
from helpers.data_handling import DataHandler

import numpy as np
import random
import os
import pickle
from time import time

from tensorflow.python.keras import backend as be
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import RNN, GRU, LSTM, Dense, Activation, Bidirectional, Masking, Embedding
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop

n_items = 1477
embedding_size = 8
max_length = 60
layers = [50]  # one LSTM layer with 50 hidden neurons
active_f = 'tanh'  # activation for rnn
batch_size = 64

learning_rate = 0.1
input_type = 'float32'

metrics = {'recall': {'direction': 1},
           'precision': {'direction': 1},
           'sps': {'direction': 1},
           'user_coverage': {'direction': 1},
           'item_coverage': {'direction': 1},
           'ndcg': {'direction': 1},
           'blockbuster_share': {'direction': -1}
           }


# class RNNOneHotK(object):

def prepare_networks(n_items, embedding_size, max_length):
    if be.backend() == 'tensorflow':
        import tensorflow as tf
        from tensorflow.keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = self.tf_mem_frac
        set_session(tf.Session(config=config))

        model = Sequential()
        if embedding_size > 0:
            model.add(Embedding(n_items, embedding_size, input_length=max_length))
            model.add(Masking(mask_value=0.0))
        else:
            model.add(Masking(mask_value=0.0, input_shape=(max_length, n_items)))

    rnn = LSTM

    for i, h in enumerate(layers):
        if i != len(layers) - 1:
            model.add(rnn(h, return_sequences=True, activation=active_f))
        else:  # last rnn return only last output
            model.add(rnn(h, return_sequences=False, activation=active_f))
    model.add(Dense(n_items))
    model.add(Activation('softmax'))

    # optimizer = self.updater()
    # def gpu_diag_wide(X):
    #     E = be.eye(*X.shape)
    #     return be.sum(X * E, axis=1)
    #
    # def bpr(yhat):
    #     return be.mean(-be.log(be.sigmoid(tf.expand_dims(gpu_diag_wide(yhat), 1) - yhat)))
    #
    # def identity_loss(y_true, y_pred):
    #     return y_true, y_pred

    # def customLoss(y_true, y_pred):
    #     # target_index = be.argmax(y_true)
    #     target_index = np.argmax(y_true)
    #     # target_index = be.get_value(target_index)
    #     y_true_pred = y_pred[0][target_index]
    #     y_true_pred = be.reshape(y_true_pred, [1, ])
    #     y_true_vec = tf.expand_dims(y_true_pred, 1)
    #     # r_uij = y_true_vec - be.transpose(y_pred)
    #     r_uij = y_true_vec - y_pred
    #     return be.mean(-be.log(be.sigmoid(r_uij)))
    
    # model.compile(loss='categorical_crossentropy', optimizer=Adagrad(lr=learning_rate))
    model.compile(loss=customLoss, optimizer=Adagrad(lr=learning_rate))
    # model.layers[-1].output
    # print(model.layers[-1].output.shape)
    # model.compile(loss=bpr(model.layers[-1].output), optimizer=Adagrad(lr=learning_rate))
    return model

def prepare_input(sequences, max_length=max_length, embedding_size=embedding_size):
    """ Sequences is a list of [user_id, input_sequence, targets]
    """
    # print("_prepare_input()")
    batch_size = len(sequences)
    # print(batch_size)

    # Shape of return variables
    if embedding_size > 0:
        X = np.zeros((batch_size, max_length),
                     dtype='int')  # keras embedding requires movie-id sequence, not one-hot
    else:
        X = np.zeros((batch_size, max_length, n_items), dtype='int')  # input of the RNN
    Y = np.zeros((batch_size, n_items), dtype='int')  # output target

    for i, sequence in enumerate(sequences):
        user_id, in_seq, target = sequence

        if embedding_size > 0:
            X[i, :len(in_seq)] = np.array([item[0] for item in in_seq])
        else:
            seq_features = np.array(list(map(lambda x: _get_features(x, n_items), in_seq)))
            X[i, :len(in_seq), :] = seq_features  # Copy sequences into X

        Y[i][target[0]] = 1.
    return X, Y


# def gen_mini_batch(sequence_generator, batch_size=64, iter=False, test=False, max_reuse_sequence=3):
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
#     # iterations = 0
#     while True:
#         j = 0
#         sequences = []
#         batch_size = batch_size
#         if test:
#             batch_size = 1
#         while j < batch_size:  # j : # of precessed sequences for rnn input
#
#             sequence, user_id = next(sequence_generator)
#             # print(user_id, len(sequence))
#
#             # finds the lengths of the different subsequences
#             # if len(sequence) <= max_length+1:  # training set
#             #     sequence = sequence[0:-1]
#             #     target = sequence[-1]
#             #     sequences.append([user_id, sequence, target])
#             # else:
#             #     sequence = sequence[-max_length-1:-1]
#             #     target = sequence[-1]
#             #     sequences.append([user_id, sequence, target])
#
#             seq_lengths = sorted(
#                 random.sample(range(2, len(sequence)),  # range, min_length = 5
#                               min([batch_size - j, len(sequence) - 2, len(sequence)//10])))
#             # always only take len(sequence)//10 sub sequences from each user
#
#             # seq_lengths = sorted(
#             #     random.sample(range(2, len(sequence)),  # range, min_length = 5
#             #                   min([batch_size - j, len(sequence) - 2])))
#             start_l = []
#             skipped_seq = 0
#             for l in seq_lengths:
#                 # target is only for rnn with hinge, logit and logsig.
#                 target = sequence[l:][0]
#                 if len(target) == 0:
#                     skipped_seq += 1
#                     continue
#                 start = max(0, l - max_length)  # sequences cannot be longer than self.max_length
#                 start_l.append(start)
#                 # print(target)
#                 sequences.append([user_id, sequence[start:l], target])
#             print(user_id, len(sequence), seq_lengths, start_l)
#
#
#             # print(user_id, len(sequence))
#             # for l in seq_lengths:
#             #     # target is only for rnn with hinge, logit and logsig.
#             #     start = np.random.randint(0, len(sequence)) # randomly choose a start position
#             #     start = min(start, len(sequence)-l-1)
#             #
#             #     target = sequence[start + l:][0]
#             #
#             #     if len(target) == 0:
#             #         skipped_seq += 1
#             #         continue
#             #     # start = max(0, l - max_length)  # sequences cannot be longer than self.max_length
#             #     # print(target)
#             #     sequences.append([user_id, sequence[start:start + l], target])
#             # print([user_id, sequence[start:l], target])
#
#             # j += 1
#             j += len(seq_lengths) - skipped_seq
#         # print(j, len(sequences), sequences[0])
#         # iterations += 1
#         # print('generating mini_batch ({})'.format(iterations))
#         # yield prepare_input(sequences)
#         return sequences


def gen_mini_batch(n_users, sequence_generator, test=False):
    ''' Takes a sequence generator and produce a mini batch generator.
    The mini batch have a size defined by self.batch_size, and have format of the input layer of the rnn.

    test determines how the sequence is splitted between training and testing
        test == False, the sequence is split randomly
        test == True, the sequence is split in the middle

    if test == False, max_reuse_sequence determines how many time a single sequence is used in the same batch.
        with max_reuse_sequence = inf, one sequence will be used to make the whole batch (if the sequence is long enough)
        with max_reuse_sequence = 1, each sequence is used only once in the batch
    N.B. if test == True, max_reuse_sequence = 1 is used anyway
    '''
    i = 0
    uid = []
    sequences = []
    j = 0 # number of user
    # dataset = dataset.training_set
    for user in range(n_users):
        sequence, user_id = next(sequence_generator)
        uid.append(user_id)

        # finds the lengths of the different subsequences
        if not test:  # training set
            seq_lengths = sorted(
                random.sample(range(2, len(sequence)),  # range, min_sub_seq_length = 10 or 2, same total amount
                              len(sequence) // 10))  # number of sub sequences generated for each user
            # seq_lengths = sorted(
            #     random.sample(range(2, len(sequence)),  # range, min_sub_seq_length = 10
            #                   len(sequence) - 2 ))

        else:  # validating set
            seq_lengths = [int(len(sequence)-1)]  # not iterate

        start_l = []
        skipped_seq = 0
        for l in seq_lengths:
            # target is only for rnn with hinge, logit and logsig.
            target = sequence[l:][0]
            if len(target) == 0:
                skipped_seq += 1
                continue
            start = max(0, l - max_length)  # sequences cannot be longer than self.max_length
            start_l.append(start)
            # print(target)
            sequences.append([user_id, sequence[start:l], target])
            print(user_id, len(sequence), seq_lengths, start_l)
        i +=1
        # print(user_id, len(sequence), seq_lengths, start_l)
        j += len(seq_lengths) - skipped_seq
    print(j)
    ########   random start
        # for l in seq_lengths:
        #     # target is only for rnn with hinge, logit and logsig.
        #     start = np.random.randint(0, len(sequence)) # randomly choose a start position
        #     start = min(start, len(sequence)-l-1)
        #     start_l.append(start)
        #     target = sequence[start + l:][0]
        #     if len(target) == 0:
        #         skipped_seq += 1
        #         continue
        #     # start = max(0, l - max_length)  # sequences cannot be longer than self.max_length
        #     # print(target)
        #     sequences.append([user_id, sequence[start:start + l], target])
        #     print(user_id, len(sequence), seq_lengths, start_l)
        # j += len(seq_lengths) - skipped_seq
        # print(j)
    ########     end here
    # iterations += 1
    # print('generating mini_batch ({})'.format(iterations))
    # yield prepare_input(sequences)
    return sequences

dataset = DataHandler(dirname="ks-cooks-1y")

# model = prepare_networks(dataset.n_items, embedding_size, max_length)

# loss = train(model, dataset)
n_users = dataset.training_set.n_users
n_val_users = dataset.validation_set.n_users

train_generator = gen_mini_batch(n_users, dataset.training_set(max_length=850))
# val_generator = gen_mini_batch(n_val_users, dataset.training_set(max_length=850), test = True)

path = '/Users/xun/Documents/Thesis/Improving-RNN-recommendation-model/Dataset/'
dirname = 'ks-cooks-1y'
# with open(path+dirname+'/data/sub_sequences_list_10.pickle', 'wb') as fp:
#     pickle.dump(train_generator, fp)

# with open(path+dirname+'/data/validation_list_10.pickle', 'wb') as fp:
#     pickle.dump(val_generator, fp)

# train_generator = gen_mini_batch(dataset.training_set(max_length=800), batch_size=64) #, batch_size=256
# val_generator = gen_mini_batch(dataset.validation_set())

# result = model.fit_generator(generator=train_generator, epochs=1, steps_per_epoch=60,  # 15100/256
#                              # validation_data=val_generator, validation_steps=1,
#                             verbose=1)
#
# result = model.fit(X, Y, epochs=1, batch_size = 256,
#                              # validation_data=val_generator, validation_steps=1,
#                              verbose=1)
# print(result.history)
#
# model.save('/Users/xun/Documents/Thesis/Improving-RNN-recommendation-model/Dataset/ks-cooks-1y/models/rnn-long-train.h5')

# """
