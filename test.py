'''
@Author: your name
@Date: 2020-05-12 20:02:56
@LastEditTime: 2020-05-12 21:13:47
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

from data_load import get_batch
from model import Transformer
from hparams import Hparams
from utils import get_hypotheses, calc_bleu, postprocess, load_hparams
import logging

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.ckpt)

logging.info("# Prepare test batches")
test_batches, num_test_batches, num_test_samples  = get_batch(hp.test_features_path, hp.test_labels_path,100000,hp.test_batch_size,
                                              shuffle=False)
iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
sparse_features, dense_features, labels = iter.get_next()
m = Transformer(hp)
pred_age, pred_gender = m.infer(sparse_features, dense_features, labels)
test_init_op = iter.make_initializer(test_batches)

logging.info("# Load model")
m = Transformer(hp)

logging.info("# Session")
with tf.Session() as sess:
    ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    ckpt = hp.ckpt if ckpt_ is None else ckpt_ # None: ckpt is a file. otherwise dir.
    saver = tf.train.Saver()

    saver.restore(sess, ckpt)

    sess.run(test_init_op)
    cpred_age, cpred_gender = sess.run([pred_age, pred_gender])
    print(cpred_age[:5])
    print(cpred_gender[:5])

