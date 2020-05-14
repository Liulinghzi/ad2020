'''
@Author: your name
@Date: 2020-05-09 14:02:59
@LastEditTime: 2020-05-14 13:10:31
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
    seq = [float(i) for i in seq_str.split()]
    return seq


def generator_fn(product_id, product_category, advertiser_id, industry, time, click_times, target):
    for idx in range(len(product_id)):
        yield (
            (
                encode(product_id[idx]),
                encode(product_category[idx]),
                encode(advertiser_id[idx]),
                encode(industry[idx])
                ),
            (
                encode(time[idx]),
                encode(click_times[idx])
                ),
            (
                encode(target[idx])[0]
                )
                # age取值1-10， gender取值1-2
                # age_gender取值1-20
                # 1-10 gender为1  11-20gender为2
                # mod10为age
            )

def input_fn(features, labels, batch_size, target_label, shuffle=False):
    shapes = (
        ([None], [None], [None], [None]), 
        ([None], [None]),
        ()
    )
    types = (
        (tf.int32, tf.int32, tf.int32, tf.int32), 
        (tf.float32, tf.float32),
        (tf.int32)
    )
    paddings = (
        (0, 0, 0, 0), 
        (0.0, 0.0),
        (0)
    )

    if target_label == 'age':
        dataset = tf.data.Dataset.from_generator(
            generator_fn,
            output_shapes=shapes,
            output_types=types,
            args=(*features, labels[0]))  # <- arguments for generator_fn. converted to np string arrays
        
    elif target_label == 'gender':
        dataset = tf.data.Dataset.from_generator(
            generator_fn,
            output_shapes=shapes,
            output_types=types,
            args=(*features, labels[1]))  # <- arguments for generator_fn. converted to np string arrays



    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def get_batch(train_features_path, train_labels_path, target_label, maxlen, batch_size, shuffle=False):
    features = load_data(train_features_path, maxlen)
    labels = load_target(train_labels_path)
    batches = input_fn(features, labels, batch_size, target_label, shuffle=shuffle)
    num_batches = calc_num_batches(len(labels[0]), batch_size)
    return batches, num_batches, len(labels[0])
