'''
@Author: your name
@Date: 2020-05-09 14:02:59
@LastEditTime: 2020-05-09 18:32:32
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ad2020/data_load.py
'''
# -*- coding: utf-8 -*-
#/usr/bin/python3
import tensorflow as tf
import pickle
from utils import calc_num_batches

def load_data(dense_seqs_path, sparse_seqs_path, maxlen):
    with open(dense_seqs_path, 'rb') as f:
        dense_seqs = pickle.load(f)
        for d_seq in dense_seqs:
            if len(d_seq) + 1 > maxlen:
                d_seq = d_seq[:maxlen]
                
    with open(sparse_seqs_path, 'rb') as f:
        sparse_seqs = pickle.load(f)
        for s_seq in sparse_seqs:
            if len(s_seq) + 1 > maxlen:
                s_seq = s_seq[:maxlen]
    return dense_seqs, sparse_seqs

def load_target(fpath):
    with open(fpath, 'rb') as f:
        age_gender = pickle.load(f)

    return age_gender

def generator_fn(dense_seqs, sparse_seqs, age_gender):
    for idx in range(len(dense_seqs)):
        d_seq = dense_seqs[idx]
        s_seq = sparse_seqs[idx]
        age, gender = age_gender[idx]
        mask_flag = 1
        #  一个beh_seq 是[[1,1,1,1], [2,2,2,2]]的行为id
        yield (d_seq, s_seq, age, gender, mask_flag)

def input_fn(dense_seqs, sparse_seqs, age_gender, batch_size, shuffle=False):
    shapes = (
        ([None, 2], [None, 6], (), (), ())
        )
    types = (
        (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
        )
    paddings = (
        (0, 0, 0, 0, 0)
        )

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(dense_seqs, sparse_seqs, age_gender))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def get_batch(dense_seqs_path, sparse_seqs_path, age_gender_path, maxlen, batch_size, shuffle=False):
    dense_seqs, sparse_seqs = load_data(dense_seqs_path, sparse_seqs_path, maxlen)
    print('load dara')
    exit()
    age_gender = load_target(age_gender_path)
    # 这里的behavior_seqs需要时已经构建好的list [[1,1,1,1], [2,2,2,2]]
    batches = input_fn(dense_seqs, sparse_seqs, age_gender, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(dense_seqs), batch_size)
    return batches, num_batches, len(dense_seqs)
