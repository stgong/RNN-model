from __future__ import print_function

import glob
import os
import re
import sys
import time
import pickle
import numpy as np
import tensorflow as tf

import helpers.command_parser as parse
from helpers import evaluation
from helpers.data_handling import DataHandler
from keras.models import Model, load_model, save_model
from keras import backend as be
from keras.losses import binary_crossentropy, categorical_crossentropy

def top_k_recommendations(model, sequence, max_length, embedding_size, n_items, k=10):
    ''' Receives a sequence of (id, rating), and produces k recommendations (as a list of ids)'''
    seq_by_max_length = sequence[-min(max_length, len(sequence)):]  # last max length or all
    # Prepare RNN input
    if embedding_size > 0:
        X = np.zeros((1, max_length), dtype=np.int32)  # keras embedding requires movie-id sequence, not one-hot
        X[0, :len(seq_by_max_length)] = np.array([item[0] for item in seq_by_max_length]) # item[0] already get the IDs only

    # else:
    #     X = np.zeros((1, max_length, n_items), dtype=np.int32)  # input shape of the RNN
    #     X[0, :len(seq_by_max_length), :] = np.array(
    #         list(map(lambda x: get_features(x), seq_by_max_length)))

    # Run RNN

    output = model.predict_on_batch(X)
    # filter out viewed items
    output[0][[i[0] for i in sequence]] = -np.inf

    # output = model.predict(X, batch_size = 1000)

    # find top k according to output
    # check n largest probability in output
    # output[0:1][0][output[0:1][0].argsort()[-10:][::-1]]
    recommendations = list(np.argpartition(-output[0], list(range(k)))[:k])
    rec_prob = [output[0][i] for i in recommendations]
    rec = list(tuple(zip(recommendations, rec_prob)))
    return X, output, list(np.argpartition(-output[0], list(range(k)))[:k]), rec

def pop_k_recommendations(model, sequence, max_length, embedding_size, n_items, k=10):
    ''' Receives a sequence of (id, rating), and produces k recommendations (as a list of ids)'''
    seq_by_max_length = sequence[-min(max_length, len(sequence)):]  # last max length or all
    # Prepare RNN input
    if embedding_size > 0:
        X = np.zeros((1, max_length), dtype=np.int32)  # keras embedding requires movie-id sequence, not one-hot
        X[0, :len(seq_by_max_length)] = np.array([item[0] for item in seq_by_max_length]) # item[0] already get the IDs only

    # else:
    #     X = np.zeros((1, max_length, n_items), dtype=np.int32)  # input shape of the RNN
    #     X[0, :len(seq_by_max_length), :] = np.array(
    #         list(map(lambda x: get_features(x), seq_by_max_length)))

    # Run RNN

    # output = model.predict_on_batch(X)
    training_set_item_popularity = np.load('/Users/xun/Documents/Thesis/Improving-RNN-recommendation-model/Dataset/ks-likes-1y-v2/data/training_set_item_popularity.npy', allow_pickle=True)
    top_10 = training_set_item_popularity.argsort()[-11:][::-1]
    output = top_10
    # filter out viewed items
    # output[0][[i[0] for i in sequence]] = -np.inf

    # output = model.predict(X, batch_size = 1000)

    # find top k according to output
    # check n largest probability in output
    # output[0:1][0][output[0:1][0].argsort()[-10:][::-1]]
    # recommendations = list(np.argpartition(-output[0], list(range(k)))[:k])
    # rec_prob = [output[0][i] for i in recommendations]
    # rec = list(tuple(zip(recommendations, rec_prob)))
    return output



n_items = 1507
embedding_size = 100
max_length = 60
metrics = {'recall': [],
           'precision': [],
           'sps': [],
           'user_coverage':[],
           'item_coverage': [],
           'ndcg': [],
           'blockbuster_share': []
           }
path = '/Users/xun/Documents/Thesis/Improving-RNN-recommendation-model/Dataset/'
dirname= "ks-likes-1y-v2"
# dirname= "ml-1M"
# model_name = 'rnn_cce_ml30_bs64_ne50.0_gc100_e8_h50_Ug_lr0.1_nt1.ktf'
model_name = 'rnn_cce_ml60_bs128_ne1899.935_gc100_e100_h100_Ug_lr0.1_nt1.ktf' #ks-cooks-2y
# model_name = 'rnn_cce_ml30_bs64_ne28.337_gc100_e8_h50_Ug_lr0.1_nt1.ktf'
dataset = DataHandler(dirname=dirname)

model = load_model(path + dirname + '/models/ktf/'+ model_name)

# model.summary()
#
# for layer in model.layers:
#    print(layer.name, layer.get_weights(), layer.output_shape)
target = []
rec = []
rec_l = []
test_u_id = []
rec_dict = {}
score = {}
true_positive = {}
test_input = []
test_output = []
index = 0
ev = evaluation.Evaluator(dataset, k=10)

for sequence, user_id in dataset.test_set(epochs=1):
    # if user_id == '2309':
    #     # print(len(sequence))
    #     # sequence = sequence[0:-5]
    #     # print(len(sequence))
    #     num_viewed = int(len(sequence) / 2)
    #     viewed = sequence[:num_viewed]
    #     viewed = viewed[0:-2]
    #     print(viewed)
    #     goal = [i[0] for i in sequence[num_viewed:]]
    #     input, output, recommendations, rec_prob = top_k_recommendations(model, viewed, max_length, embedding_size, 10)
    #     print(rec_prob)
    #     for item in recommendations:
    #         if item in goal:
    #             true_positive[user_id].append(item)
    #     print(true_positive)
    num_viewed = int(len(sequence) / 2)
    viewed = sequence[:num_viewed]
    # print(viewed)
    goal = [i[0] for i in sequence[num_viewed:]]  # list of movie ids

    input, output, recommendations, rec_prob = top_k_recommendations(model, viewed, max_length, embedding_size, 10)
    input, output, recommendations, rec_prob = input, output, recommendations, rec_prob

    recommendations_pop10 = pop_k_recommendations(model, viewed, max_length, embedding_size,10)

    # with open(path+dirname + '/data/implicit_predictions.pickle', 'rb') as fp:
    #     implicit_predictions = pickle.load(fp)
    # recommendations = [item[0] for item in implicit_predictions[index]]

    ##################### save some data for detail review ###################
    '''
    #get a list of whole target ids for all the users in order to count the distinct items of the target parts in testset
    #get a list of whole recommended ids for all the users in order to count the distinct recommended items of input model
    '''
    for item in goal:
        target.append(item)
    for item in recommendations:
        rec_l.append(item)

    rec.append(rec_prob)

    true_positive[user_id]=[]
    for item in recommendations:
        if item in goal:
            true_positive[user_id].append(item)

    test_u_id.append(user_id)

    # rec_dict[user_id] = recommendations
    rec_dict[user_id] = recommendations

    #
    # for item in viewed:
    #     test_input.append([user_id, item[0]])
    #
    # for item in goal:
    #     test_output.append([user_id, item])
    ev_i = evaluation.Evaluator(dataset, k=10)
    ev_i.add_instance(goal, recommendations_pop10)
    ev.add_instance(goal, recommendations_pop10)

    metrics['recall'].append(ev_i.average_recall())
    metrics['sps'].append(ev_i.sps())
    metrics['precision'].append(ev_i.average_precision())
    metrics['ndcg'].append(ev_i.average_ndcg())
    metrics['user_coverage'].append(ev_i.user_coverage())
    metrics['item_coverage'].append(ev_i.item_coverage())
    metrics['blockbuster_share'].append(ev_i.blockbuster_share())

    score[user_id] = [metrics['sps'][-1], metrics['precision'][-1]]
    ##################### check the output of the cost function ###################
    # Y = np.zeros((1, n_items), dtype='int')
    # Y[0][goal[0]] = 1.
    #
    # test_loss = model.test_on_batch(input, Y)
    #
    # y_true = tf.convert_to_tensor(Y, np.float32)
    # y_prep = tf.convert_to_tensor(output, np.float32)
    #
    # # error_1 = K.eval(binary_crossentropy(y_true, y_prep))
    # # error_2 = K.eval(categorical_crossentropy(y_true, y_prep))
    #
    #
    # def gpu_diag_wide(X):
    #     E = be.eye(*X.shape)
    #     return be.sum(X * E, axis=1)

    # def bpr(yhat):
    #     return be.mean(-be.log(be.sigmoid(tf.expand_dims(gpu_diag_wide(yhat), 1) - yhat)))

    # def bpr_loss(y_true, y_pred):
    #     diff = tf.expand_dims(y_pred[0], 1) - y_pred[0]
    #     # sig = be.sigmoid(diff)
    #     sig = be.square(diff)
    #     log_loss = - be.log(sig)
    #     log_loss_mean = be.mean(log_loss, axis=1)
    #     loss = tf.tensordot(y_true, log_loss_mean, axes=1)
    #     return loss
    # cost1 = be.eval(bpr(output[0]))
    # cost2 = be.eval(bpr_loss(y_true,y_prep))
    # print(cost1)
    ############################################################################

    ##########################Intermediate result print#########################
    # intermediate_model = Model(inputs=model.layers[0].input,
    #                            outputs=[l.output for l in model.layers[1:]])
    # intermediate_output = intermediate_model.predict(input)
    # print(intermediate_output)
    ##########################Intermediate result print#########################

    # print(user_id, recommendations)
    # print(user_id, rec_prob, metrics['sps'][-1], metrics['precision'][-1])
    # ev_i.instances = []
    index +=1
print(len(set(target)), len(set(rec_l)))


ev.nb_of_dp = dataset.n_items
metrics_t = 'sps,recall,precision,item_coverage,user_coverage,ndcg,blockbuster_share'
metrics_t = metrics_t.split(',')
for m in metrics_t:
    if m not in ev.metrics:
        raise ValueError('Unknown metric: ' + m)

    print(m + '@' + str(ev.k) + ': ', ev.metrics[m]())

# outfile = path+dirname+'/data/test_u_id_emb16.pickle'
# outfile1 = path+dirname+'/data/rec_dict_emb16.pickle'
# outfile2 = path+dirname+'/data/score_dict_emb16.pickle'
# outfile3 = path+dirname+'/data/true_positive_emb16.pickle'
# # outfile4 = path+dirname+'/data/true_positive_imp.pickle'
# outfile5 = path+dirname+'/data/rnn_predictions_emb16.pickle'
# with open(outfile, 'wb') as fp:
#     pickle.dump(test_u_id, fp)
# with open(outfile1, 'wb') as fp:
#     pickle.dump(rec_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
# with open(outfile2, 'wb') as fp:
#     pickle.dump(score, fp, protocol=pickle.HIGHEST_PROTOCOL)
# with open(outfile3, 'wb') as fp:
#     pickle.dump(true_positive, fp, protocol=pickle.HIGHEST_PROTOCOL)
# with open(outfile5, 'wb') as fp:
#     pickle.dump(rec, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open(path+dirname+'/data/test_input.pickle', 'wb') as fp:
#     pickle.dump(test_input, fp)
# with open(path+dirname+'/data/test_output.pickle', 'wb') as fp:
#     pickle.dump(test_output, fp)


