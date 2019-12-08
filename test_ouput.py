from __future__ import print_function

from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

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
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.python.keras import backend as be
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy


n_items = 1477
embedding_size = 8
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
dirname= "ks-cooks-2y"
# model_name = 'rnn_cce_ml30_bs64_ne50.0_gc100_e8_h50_Ug_lr0.1_nt1.ktf'
# model_name = 'rnn_cce_ml60_bs32_ne2406.162_gc100_e16_h50_Ug_lr0.1_nt1.ktf'
dataset = DataHandler(dirname=dirname)

test_u_id = []
for sequence, user_id in dataset.test_set(epochs=1):
    test_u_id.append(user_id)

outfile = path+dirname+'/data/test_u_id.pickle'
with open(outfile, 'wb') as fp:
    pickle.dump(test_u_id, fp)