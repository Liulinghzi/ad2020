'''
@Author: your name
@Date: 2020-05-09 14:02:59
@LastEditTime: 2020-05-12 21:15:33
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ad2020/test.py
'''
# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Inference
'''

import os

import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch
from model import Transformer
from hparams import Hparams
from utils import load_hparams
import logging

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.ckpt)

logging.info("# Prepare test batches")
test_batches, num_test_batches, num_test_samples  = get_batch(
    train_features_path=hp.test_features_path, 
    train_labels_path=hp.test_labels_path, 
    maxlen=10000, 
    batch_size=hp.test_batch_size, 
    shuffle=False
    )

iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
sparse_features, dense_features, labels = iter.get_next()

test_init_op = iter.make_initializer(test_batches)

logging.info("# Load model")
m = Transformer(hp)
pred_age, pred_gender = m.infer(sparse_features, dense_features)

logging.info("# Session")
with tf.Session() as sess:
    ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    ckpt = hp.ckpt if ckpt_ is None else ckpt_ # None: ckpt is a file. otherwise dir.
    saver = tf.train.Saver()

    saver.restore(sess, ckpt)
        
    sess.run(test_init_op)
    predicted_age = []
    predicted_gender = []
    
    for i in tqdm(range(num_test_batches)):
        cpred_age, cpred_gender = sess.run([pred_age, pred_gender])
        predicted_age.extend(cpred_age.tolist())
        predicted_gender.extend(cpred_gender.tolist())
    print(len(predicted_age))


    import pandas as pd
    import pickle
    with open('/home/liulingzhi1/notespace/ctr_practice/tencent/5-8/test_user_id.pkl', 'rb') as f:
        user_id = pickle.load(f)

    submit =pd.DataFrame(
        {
            # 'user_id':user_id,
            'predicted_age': predicted_age,
            'predicted_gender':predicted_gender,
        })

    submit.to_csv('submit.csv', index=False)


