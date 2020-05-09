# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf

from data_load import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging
import pickle

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp
        with open(self.hp.vocab_list, 'rb') as f:
            self.hp.vocab_list = pickle.load(f)
        self.embedding_list = [get_token_embeddings(self.hp.vocab_dict[i], self.hp.d_model, zero_pad=True) for i in range(len(self.hp.vocab_list))]
        '''
        这里就不只是一个embeddings了而是
        self.embedding_dict = {feat: get_token_embeddings for feat in features}
        '''

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens = xs

            # src_masks
            src_masks = tf.math.equal(x, 0) # (N, T1)

            # embedding
            # x的形状为[
            #     [[1,1,1,1],[2,2,2,1]],
            #     [[1,1,1,1],[2,2,2,1]],
            #     [[1,1,1,1],[2,2,2,1]],
            #     [[1,1,1,1],[2,2,2,1]],
            # ]
            # [bs, maxlen, feat_num]
            # 需要在feat_num的维度进行切分，分别lookup，再FM和concate，FM以后在加
            splited_x = tf.split(axis=2, value=x, num_split=x.shape[2])
            splited_x = [tf.reshape(xi, (-1, xi.shape[1])) for xi in splited_x]

            encs = [tf.nn.embedding_lookup(self.embedding_list[i], x[i]) for i in range(len(splited_x))]
            # lookup后每个encs应该是 [[emb][emb]]

            enc = tf.concat(concat_dim=1, values=encs)
            # concate之后应该是[con_emb, con_emb]
            
            # enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale
            # 这里的enc需要从embedding_dict中的多个matrix中lookup，然后concat，作为一个enc

            enc += positional_encoding(enc, self.hp.maxlen)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)
            age_enc = enc
            gender_enc = enc
            
            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_age_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    age_enc = multihead_attention(queries=age_enc,
                                              keys=age_enc,
                                              values=age_enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    age_enc = ff(age_enc, num_units=[self.hp.d_ff, self.hp.d_model])

                with tf.variable_scope("num_blocks_gender_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    gender_enc = multihead_attention(queries=gender_enc,
                                              keys=gender_enc,
                                              values=gender_enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    gender_enc = ff(gender_enc, num_units=[self.hp.d_ff, self.hp.d_model])

        age_logits = tf.layers.dense(age_enc, self.hp.age_classes)        
        gender_logits = tf.layers.dense(gender_enc, self.hp.gender_classes)        
        
        return age_logits, gender_logits, src_masks

    def train(self, xs, y_age, y_gender):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        age_logits, gender_logits, src_masks = self.encode(xs)

        # train scheme
        y_age_ = label_smoothing(tf.one_hot(y_age, depth=self.hp.age_classes))
        y_gender_ = label_smoothing(tf.one_hot(y_gender, depth=self.hp.gender_classes))
        
        ce_age = tf.nn.softmax_cross_entropy_with_logits_v2(logits=age_logits, labels=y_age_)
        ce_gender = tf.nn.softmax_cross_entropy_with_logits_v2(logits=gender_logits, labels=y_age_)
        
        # loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
        loss = ce_gender + ce_age

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    # def eval(self, xs, ys):
    #     '''Predicts autoregressively
    #     At inference, input ys is ignored.
    #     Returns
    #     y_hat: (N, T2)
    #     '''
    #     memory, sents1, src_masks = self.encode(xs, False)

    #     logging.info("Inference graph is being built. Please be patient.")
    #     for _ in tqdm(range(self.hp.maxlen2)):
    #         logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
    #         if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

    #         _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
    #         ys = (_decoder_inputs, y, y_seqlen, sents2)

    #     # monitor a random sample
    #     n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
    #     sent1 = sents1[n]
    #     pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
    #     sent2 = sents2[n]

    #     tf.summary.text("sent1", sent1)
    #     tf.summary.text("pred", pred)
    #     tf.summary.text("sent2", sent2)
    #     summaries = tf.summary.merge_all()

    #     return y_hat, summaries

