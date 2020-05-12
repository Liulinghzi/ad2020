'''
@Author: your name
@Date: 2020-05-09 14:02:59
@LastEditTime: 2020-05-12 12:39:13
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
    with open(train_features_path, 'rb') as f:
        features = pickle.load(f)

    return features

def load_target(train_labels_path):
    with open(train_labels_path, 'rb') as f:
        labels = pickle.load(f)

    return labels


def encode(seq_str):
    seq = [float(i) for i in seq_str.split(',')]
    return seq


def generator_fn(creative_id, ad_id, product_id, product_category, advertiser_id, industry, time, click_times, age, gender):

    for idx in range(len(creative_id)):
        yield (
            (
                encode(creative_id[idx]),
                encode(ad_id[idx]),
                encode(product_id[idx]),
                encode(product_category[idx]),
                encode(advertiser_id[idx]),
                encode(industry[idx])
                ),
            (
                encode(time[idx]),
                encode(click_times[idx])
                )
            (
                encode(age[idx]),
                encode(gender[idx])
                )
            )

def input_fn(features, labels, batch_size, shuffle=False):
    shapes = (
        ([None], [None], [None], [None], [None], [None]), 
        ([None], [None]),
        ([], [])
    )
    types = (
        (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32), 
        (tf.float32, tf.float32),
        (tf.int32, tf.int32)
    )
    paddings = (
        (0, 0, 0, 0, 0, 0), 
        (0.0, 0.0),
        (0, 0)
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
    creative_id = features[0]
    for idx in range(len(creative_id)):
        print(encode(creative_id[idx]))
    # 这里的behavior_seqs需要时已经构建好的list [[1,1,1,1], [2,2,2,2]]
    batches = input_fn(features, labels, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(labels[0]), batch_size)
    return batches, num_batches, len(labels[0])
