'''
@Author: your name
@Date: 2020-05-09 14:02:59
@LastEditTime: 2020-05-12 10:23:31
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ad2020/data_load.py
'''
# -*- coding: utf-8 -*-
#/usr/bin/python3
import tensorflow as tf
import pickle
import numpy as np
from utils import calc_num_batches

def load_data(train_features_path, maxlen):
    res = []
    with open(train_features_path, 'rb') as f:
        features = pickle.load(f)
        for feat in features:
            for seq in feat:
                if len(seq) + 1 > maxlen:
                    seq = seq[:maxlen]
            res.append(feat)

    return res

def load_target(train_labels_path):
    with open(train_labels_path, 'rb') as f:
        labels = pickle.load(f)

    return labels

def generator_fn(creative_id, ad_id, product_id, product_category, advertiser_id, industry, time, click_times, age, gender):
    for idx in range(len(creative_id)):
        yield (
            creative_id[idx], ad_id[idx], product_id[idx], product_category[idx], advertiser_id[idx], industry[idx], 
            time[idx], click_times[idx], 
            age, gender
            )

def input_fn(features, labels, batch_size, shuffle=False):
    shapes = (
        (
            [None], [None], [None], [None], [None], [None], 
            [None], [None],
            (), ()
        )
    )
    types = (
        (
            tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, 
            tf.float32, tf.float32,
            tf.int32, tf.int32)
        )
    paddings = (
        (
            0, 0, 0, 0, 0, 0, 
            0.0, 0.0,
            0, 0)
        )
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(*features, *labels))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def get_batch(train_features_path, train_labels_path, maxlen, batch_size, shuffle=False):
    features = load_data(train_features_path, maxlen)
    labels = load_target(train_labels_path)
    # 这里的behavior_seqs需要时已经构建好的list [[1,1,1,1], [2,2,2,2]]
    batches = input_fn(features, labels, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(dense_seqs), batch_size)
    return batches, num_batches, len(dense_seqs)
