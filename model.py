# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging
import pickle

logging.basicConfig(level=logging.INFO)

class Transformer:
    def __init__(self, hp):
        self.hp = hp

        with open(self.hp.vocab_list, 'rb') as f:
            self.hp.vocab_list = pickle.load(f)
        # 前两个特征是dense，1维
        # self.embedding_creative_id = get_token_embeddings(self.hp.vocab_list[0], self.hp.d_model, 'creative_id', zero_pad=True)
        # self.embedding_ad_id = get_token_embeddings(self.hp.vocab_list[1], self.hp.d_model, 'ad_id', zero_pad=True)
        self.embedding_product_id = get_token_embeddings(self.hp.vocab_list[0], self.hp.d_model, 'product_id', self.hp, zero_pad=True)
        self.embedding_product_category = get_token_embeddings(self.hp.vocab_list[1], self.hp.d_model, 'product_category', self.hp, zero_pad=True)
        self.embedding_advertiser_id = get_token_embeddings(self.hp.vocab_list[2], self.hp.d_model, 'advertiser_id', self.hp, zero_pad=True)
        self.embedding_industry = get_token_embeddings(self.hp.vocab_list[3], self.hp.d_model, 'industry', self.hp, zero_pad=True)
        '''
        这里就不只是一个embeddings了而是
        self.embedding_dict = {feat: get_token_embeddings for feat in features}
        '''

    def encode(self, sparse_features, dense_features, labels=None, training=True, target_label='age'):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            product_id, product_category, advertiser_id, industry = sparse_features
            time, click_times = dense_features
            target = labels

            # src_masks
            # 在get_batch的时候用了0来作为pad
            src_masks = tf.math.equal(product_id, 0) # (N, T1)

            product_id_enc = tf.nn.embedding_lookup(self.embedding_product_id, product_id)
            product_category_enc = tf.nn.embedding_lookup(self.embedding_product_category, product_category)
            advertiser_id_enc = tf.nn.embedding_lookup(self.embedding_advertiser_id, advertiser_id)
            industry_enc = tf.nn.embedding_lookup(self.embedding_industry, industry)


            # enbedding的维度是[bs, seqlen, embedding]
            # dense_features的维度是[bs, seqlen]
            encs = [product_id_enc, product_category_enc, advertiser_id_enc, industry_enc]
            print('product_id_enc', product_id_enc.shape)
            print('product_category_enc', product_category_enc.shape)
            print('advertiser_id_enc', advertiser_id_enc.shape)
            print('industry_enc', industry_enc.shape)

            for enc in encs:
                enc *= self.hp.d_model**0.5 # scale
                enc += positional_encoding(enc, self.hp.maxlen)
                enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            time = tf.expand_dims(time, -1)
            click_times = tf.expand_dims(click_times, -1)
            print('time', time.shape)
            print('click_times', click_times.shape)


            encs += [time, click_times]

            concated_enc = tf.concat(axis=2, values=encs)
            # concate之后应该是[con_emb, con_emb]
            

            # 这里的enc需要从embedding_dict中的多个matrix中lookup，然后concat，作为一个enc

            target_enc = concated_enc
            print('target_enc', target_enc.shape)
            
            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_%s_%d" % (target_label, i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    target_enc = multihead_attention(queries=target_enc,
                                              keys=target_enc,
                                              values=target_enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    print('target_enc', target_enc.shape)
                    # feed forward
                    target_enc = ff(target_enc, num_units=[self.hp.d_ff, target_enc.shape[-1]])


        target_enc = tf.reduce_sum(target_enc, axis=1)
        print('target_enc', target_enc.shape)
        if target_label == 'age':
            target_logits = tf.layers.dense(target_enc, self.hp.age_classes)        
        elif target_label == 'gender':
            target_logits = tf.layers.dense(target_enc, self.hp.gender_classes)        
        
        return target_logits, src_masks

    def train(self, sparse_features, dense_features, labels, target_label):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        
        # forward
        target_logits, src_masks = self.encode(sparse_features, dense_features, labels, target_label=target_label)
        target = labels

        # train scheme
        if target_label == 'age':
            target_ = label_smoothing(tf.one_hot(target, depth=self.hp.age_classes))
        elif target_label == 'gender':
            target_ = label_smoothing(tf.one_hot(target, depth=self.hp.gender_classes))
        
        ce_target = tf.nn.softmax_cross_entropy_with_logits_v2(logits=target_logits, labels=target_)
        
        # loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
        loss = tf.reduce_mean(ce_target)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, sparse_features, dense_features, labels, target_label):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''

        target_logits, src_masks = self.encode(sparse_features, dense_features, labels, target_label=target_label)
        target = labels

        target_ = label_smoothing(tf.one_hot(target, depth=self.hp.age_classes*self.hp.gender_classes))
        
        ce_target = tf.nn.softmax_cross_entropy_with_logits_v2(logits=target_logits, labels=target_)
        
        # loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
        loss = tf.reduce_mean(ce_target)


        logging.info("Inference graph is being built. Please be patient.")
        pred_target = tf.argmax(target_logits, axis=1) + 1
        # monitor a random sample

        acc = tf.metrics.accuracy(labels=labels, predictions=pred_target)
        tf.summary.scalar("eval_loss", loss)
        tf.summary.scalar("eval_loss", acc)
        summaries = tf.summary.merge_all()

        return pred_target, target, summaries

    def infer(self, sparse_features, dense_features, target_label):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''

        target_logits, src_masks = self.encode(sparse_features, dense_features, target_label)

        pred_target = tf.argmax(target_logits, axis=1) + 1

        return pred_target

