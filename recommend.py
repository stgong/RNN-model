from __future__ import print_function

import glob
import os
import re
import sys
import time
import pickle
import numpy as np
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import helpers.command_parser as parse
from helpers import evaluation
from helpers.data_handling import DataHandler
from tensorflow.keras.models import Model, load_model, save_model
from keras import backend as be
from keras.losses import binary_crossentropy, categorical_crossentropy

def top_k_recommendations(model, sequence, max_length, embedding_size, n_items, k=10):
    ''' Receives a sequence of (id, rating), and produces k recommendations (as a list of ids)'''
    seq_by_max_length = sequence[-min(max_length, len(sequence)):]  # last max length or all
    # Prepare RNN input
    if embedding_size > 0:
        X = np.zeros((1, max_length), dtype=np.int32)  # keras embedding requires movie-id sequence, not one-hot
        X[0, :len(seq_by_max_length)] = np.array([item[0] for item in seq_by_max_length]) # item[0] already get the IDs only

    output = model.predict_on_batch(X)
    # filter out viewed items
    output[0][[i[0] for i in sequence]] = -np.inf

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
    training_set_item_popularity = np.load('/Users/xun/Documents/Thesis/RNN-model/ks-cooks-1y/data/training_set_item_popularity.npy', allow_pickle=True)
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



n_items = 1477
embedding_size = 100
max_length = 60
metrics = {
            'recall':[],
            'precision':[],
            'sps':[],
            'sps_short':[],
            'sps_long':[],
            'user_coverage':[],
            'item_coverage':[],
            'total_item_coverage':[],
            'uniq_rec':[],
            'ndcg':[],
            'blockbuster_share':[],
            'intra_list_similarity':[]
           }

path = '/Users/xun/Documents/Thesis/RNN-model/'

dirname= "ks-cooks-1y"
# dirname= "ml-1M"

model_name = 'rnn_cce_ml60_bs32_ne3378.139_gc100_e100_h100_Ug_lr0.1_nt1.hdf5'

dataset = DataHandler(dirname=dirname)

model = load_model('/Users/xun/Documents/Thesis/RNN-model/ks-cooks-1y/models/' + model_name)


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
    num_viewed = int(len(sequence) / 2)
    viewed = sequence[:num_viewed]
    # print(viewed)
    goal = [i[0] for i in sequence[num_viewed:]]  # list of movie ids

    input, output, recommendations, rec_prob = top_k_recommendations(model, viewed, max_length, embedding_size, 10)
    input, output, recommendations, rec_prob = input, output, recommendations, rec_prob

    recommendations_pop10 = pop_k_recommendations(model, viewed, max_length, embedding_size,10)

############Import the results from implicit model directly and evaluate############

    with open(path+dirname + '/data/implicit_predictions_als_h100.pickle', 'rb') as fp:
        implicit_predictions = pickle.load(fp)
    recommendations = [item[0] for item in implicit_predictions[index]]

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
    rec_dict[user_id] = recommendations

    #
    # for item in viewed:
    #     test_input.append([user_id, item[0]])
    #
    # for item in goal:
    #     test_output.append([user_id, item])

    # Here add the goal list and recommendation list
    ev_i = evaluation.Evaluator(dataset, k=10)

    ev_i.add_instance(goal, recommendations)
    ev.add_instance(goal, recommendations)

    metrics['recall'].append(ev_i.average_recall())
    metrics['sps'].append(ev_i.sps())
    # metrics['sps_short'].append(ev_i.sps_short())
    # metrics['sps_long'].append(ev_i.sps_long())
    metrics['precision'].append(ev_i.average_precision())
    metrics['ndcg'].append(ev_i.average_ndcg())
    metrics['user_coverage'].append(ev_i.user_coverage())
    metrics['total_item_coverage'].append(ev_i.total_item_coverage())
    metrics['uniq_rec'].append(ev_i.uniq_rec())
    metrics['item_coverage'].append(ev_i.item_coverage())
    metrics['blockbuster_share'].append(ev_i.blockbuster_share())
    metrics['intra_list_similarity'].append(ev_i.average_intra_list_similarity())

    score[user_id] = [metrics['sps'][-1], metrics['precision'][-1]]

    index +=1
print(len(set(target)), len(set(rec_l)))


ev.nb_of_dp = dataset.n_items
metrics_t = 'sps,sps_short,sps_long,recall,precision,uniq_rec,total_item_coverage,item_coverage,user_coverage,ndcg,' \
            'blockbuster_share,intra_list_similarity'
# sps_short,sps_long,
metrics_t = metrics_t.split(',')
for m in metrics_t:
    if m not in ev.metrics:
        raise ValueError('Unknown metric: ' + m)

    print(m + '@' + str(ev.k) + ': ', ev.metrics[m]())

# outfile = path+dirname+'/data/test_u_id.pickle'
# outfile1 = path+dirname+'/data/rec_dict_lstm.pickle'
# outfile2 = path+dirname+'/data/score_dict_lstm.pickle'
# outfile3 = path+dirname+'/data/true_positive_lstm.pickle'

# outfile4 = path+dirname+'/data/true_positive_imp_bpr_h50.pickle'
# outfile5 = path+dirname+'/data/rnn_predictions.pickle'
# with open(outfile, 'wb') as fp:
#     pickle.dump(test_u_id, fp)
# with open(outfile1, 'wb') as fp:
#     pickle.dump(rec_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
# with open(outfile2, 'wb') as fp:
#     pickle.dump(score, fp, protocol=pickle.HIGHEST_PROTOCOL)
# with open(outfile3, 'wb') as fp:
#     pickle.dump(true_positive, fp, protocol=pickle.HIGHEST_PROTOCOL)

# with open(outfile4, 'wb') as fp:
#     pickle.dump(true_positive, fp, protocol=pickle.HIGHEST_PROTOCOL)
# with open(outfile5, 'wb') as fp:
#     pickle.dump(rec, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open(path+dirname+'/data/test_input.pickle', 'wb') as fp:
#     pickle.dump(test_input, fp)
# with open(path+dirname+'/data/test_output.pickle', 'wb') as fp:
#     pickle.dump(test_output, fp)


