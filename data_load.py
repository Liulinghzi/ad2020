'''
@Author: your name
@Date: 2020-05-09 14:02:59
@LastEditTime: 2020-05-09 14:11:50
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ad2020/data_load.py
'''
# -*- coding: utf-8 -*-
#/usr/bin/python3
import tensorflow as tf
import pickle
from utils import calc_num_batches

def load_data(fpath, maxlen):
    with open(fpath, 'rb') as f:
        behavior_seqs = pickle.load(f)
        for behavior_seq in behavior_seqs:
            if len(behavior_seq) + 1 > maxlen:
                behavior_seq = behavior_seq[:maxlen]
    return behavior_seqs

def generator_fn(beh_seqs):
    for beh_seq in beh_seqs:
        #  一个beh_seq 是[[1,1,1,1], [2,2,2,2]]的行为id
        beh_seq_len = len(beh_seq)
        yield (beh_seq, beh_seq_len)

def input_fn(beh_seqs, vocab_fpath, batch_size, shuffle=False):
    shapes = (
        ([None], ())
        )
    types = (
        (tf.int32, tf.int32)
        )
    paddings = (
        (0, 0)
        )

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(beh_seqs))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def get_batch(fpath1, maxlen1, vocab_fpath, batch_size, shuffle=False):
    behavior_seqs = load_data(fpath1, maxlen1)
    # 这里的behavior_seqs需要时已经构建好的list [[1,1,1,1], [2,2,2,2]]
    batches = input_fn(behavior_seqs, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(behavior_seqs), batch_size)
    return batches, num_batches, len(behavior_seqs)
