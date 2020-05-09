# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Note.
if safe, entities on the source side have the prefix 1, and the target side 2, for convenience.
For example, fpath1, fpath2 means source file path and target file path, respectively.
'''
import tensorflow as tf
import pickle
from utils import calc_num_batches

def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''
    vocab = [line.split()[0] for line in open(vocab_fpath, 'r').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token

def load_data(fpath, maxlen):
    '''Loads source and target data and filters out too lengthy samples.
    fpath: source file path. string.
    maxlen: source sequence maximum length. scalar.

    Returns
    behavior_seqs: list of behavior sequence
    '''

    with open(fpath, 'rb') as f:
        behavior_seqs = pickle.load(f)
        for behavior_seq in behavior_seqs:
            if len(behavior_seq) + 1 > maxlen:
                behavior_seq = behavior_seq[:maxlen]
    return behavior_seqs


def encode(inp, type, dict):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''
    inp_str = inp.decode("utf-8")
    if type=="x": tokens = inp_str.split() + ["</s>"]
    else: tokens = ["<s>"] + inp_str.split() + ["</s>"]

    x = [dict.get(t, dict["<unk>"]) for t in tokens]
    return x

def generator_fn(beh_seqs, vocab_fpath):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.

    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    for beh_seq in beh_seqs:
        #  一个beh_seq 是[[1,1,1,1], [2,2,2,2]]的行为id
        beh_seq_len = len(beh_seq)
        yield (beh_seq, beh_seq_len)

def input_fn(beh_seqs, vocab_fpath, batch_size, shuffle=False):
    '''Batchify data
    beh_seqs: list of behavior seq
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
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
        args=(beh_seqs, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def get_batch(fpath1, maxlen1, vocab_fpath, batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath1: source file path. string.
    maxlen1: source sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    behavior_seqs = load_data(fpath1, maxlen1)
    # 这里的behavior_seqs需要时已经构建好的list [[1,1,1,1], [2,2,2,2]]
    batches = input_fn(behavior_seqs, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(behavior_seqs), batch_size)
    return batches, num_batches, len(behavior_seqs)
