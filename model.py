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
        self.embedding_product_id = get_token_embeddings(self.hp.vocab_list[0], self.hp.d_model, 'product_id', zero_pad=True)
        self.embedding_product_category = get_token_embeddings(self.hp.vocab_list[1], self.hp.d_model, 'product_category', zero_pad=True)
        self.embedding_advertiser_id = get_token_embeddings(self.hp.vocab_list[2], self.hp.d_model, 'advertiser_id', zero_pad=True)
        self.embedding_industry = get_token_embeddings(self.hp.vocab_list[3], self.hp.d_model, 'industry', zero_pad=True)
        '''
        这里就不只是一个embeddings了而是
        self.embedding_dict = {feat: get_token_embeddings for feat in features}
        '''

    def encode(self, sparse_features, dense_features, labels=None, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            product_id, product_category, advertiser_id, industry = sparse_features
            time, click_times = dense_features
            age = labels

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

            age_enc = concated_enc
            print('age_enc', age_enc.shape)
            
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
                    print('age_enc', age_enc.shape)
                    # feed forward
                    age_enc = ff(age_enc, num_units=[self.hp.d_ff, age_enc.shape[-1]])


        age_enc = tf.reduce_sum(age_enc, axis=1)
        print('age_enc', age_enc.shape)
        age_logits = tf.layers.dense(age_enc, 1)        
        
        return age_logits, src_masks

    def train(self, sparse_features, dense_features, labels):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        
        # forward
        age_logits, src_masks = self.encode(sparse_features, dense_features, labels)
        age = tf.expand_dims(labels, -1)

        # train scheme
        
        mse_age = tf.losses.mean_squared_error(predictions=age_logits, labels=age)
        
        # loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
        loss = tf.reduce_mean(mse_age)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, sparse_features, dense_features, labels):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''

        age_gender_logits, src_masks = self.encode(sparse_features, dense_features, labels)
        age_gender = labels

        age_gender_ = label_smoothing(tf.one_hot(age_gender, depth=self.hp.age_classes*self.hp.gender_classes))
        
        ce_age_gender = tf.nn.softmax_cross_entropy_with_logits_v2(logits=age_gender_logits, labels=age_gender_)
        
        # loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
        loss = tf.reduce_mean(ce_age_gender)


        logging.info("Inference graph is being built. Please be patient.")
        pred_age_gender = tf.argmax(age_gender_logits, axis=1)
        pred_age = tf.mod(pred_age_gender, 10)
        pred_gender = tf.ceil(tf.divide(pred_age_gender, 10))
        # monitor a random sample

        tf.summary.scalar("eval_loss", loss)
        tf.summary.text("pred_age", pred_age)
        tf.summary.text("pred_gender", pred_gender)
        summaries = tf.summary.merge_all()

        return pred_age, pred_gender, summaries

    def infer(self, sparse_features, dense_features):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''

        age_logits, src_masks = self.encode(sparse_features, dense_features)

        pred_age = tf.round(age_logits, axis=1)

        return pred_age

