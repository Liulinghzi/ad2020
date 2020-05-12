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
        self.embedding_creative_id = get_token_embeddings(self.hp.vocab_dict[0], self.hp.d_model, zero_pad=True)
        self.embedding_ad_id = get_token_embeddings(self.hp.vocab_dict[1], self.hp.d_model, zero_pad=True)
        self.embedding_product_id = get_token_embeddings(self.hp.vocab_dict[2], self.hp.d_model, zero_pad=True)
        self.embedding_product_category = get_token_embeddings(self.hp.vocab_dict[3], self.hp.d_model, zero_pad=True)
        self.embedding_advertiser_id = get_token_embeddings(self.hp.vocab_dict[4], self.hp.d_model, zero_pad=True)
        self.embedding_industry = get_token_embeddings(self.hp.vocab_dict[5], self.hp.d_model, zero_pad=True)
        '''
        这里就不只是一个embeddings了而是
        self.embedding_dict = {feat: get_token_embeddings for feat in features}
        '''

    def encode(self, sparse_features, dense_features, labels, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            creative_id, ad_id, product_id, product_category, advertiser_id, industry = sparse_features
            time, click_times = dense_features
            age, gender = labels

            # src_masks
            # 在get_batch的时候用了0来作为pad
            src_masks = tf.math.equal(creative_id, 0) # (N, T1)

            # embedding
            # x的形状为[
            #     [[1,1,1,1],[2,2,2,1]],
            #     [[1,1,1,1],[2,2,2,1]],
            #     [[1,1,1,1],[2,2,2,1]],
            #     [[1,1,1,1],[2,2,2,1]],
            # ]
            # [bs, maxlen, feat_num]
            # 需要在feat_num的维度进行切分，分别lookup，再FM和concate，FM以后在加

            creative_id_enc = tf.nn.embedding_lookup(self.embedding_creative_id, creative_id)
            ad_id_enc = tf.nn.embedding_lookup(self.embedding_ad_id, ad_id)
            product_id_enc = tf.nn.embedding_lookup(self.embedding_product_id, product_id)
            product_category_enc = tf.nn.embedding_lookup(self.embedding_product_category, product_category)
            advertiser_id_enc = tf.nn.embedding_lookup(self.embedding_advertiser_id, advertiser_id)
            industry_enc = tf.nn.embedding_lookup(self.embedding_industry, industry)


            # enbedding的维度是[bs, seqlen, embedding]
            # dense_features的维度是[bs, seqlen]
            encs = [creative_id_enc, ad_id_enc, product_id_enc, product_category_enc, advertiser_id_enc, industry_enc]
            for enc in encs:
                enc *= self.hp.d_model**0.5 # scale
                enc += positional_encoding(enc, self.hp.maxlen)
                enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            encs += [time, click_times]

            concated_enc = tf.concat(concat_dim=1, values=encs)
            # concate之后应该是[con_emb, con_emb]
            

            # 这里的enc需要从embedding_dict中的多个matrix中lookup，然后concat，作为一个enc

            age_enc = concated_enc
            gender_enc = concated_enc
            
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

    def train(self, sparse_features, dense_features, labels):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        
        # forward
        age_logits, gender_logits, src_masks = self.encode(sparse_features, dense_features, labels)

        # train scheme
        age_ = label_smoothing(tf.one_hot(age, depth=self.hp.age_classes))
        gender_ = label_smoothing(tf.one_hot(gender, depth=self.hp.gender_classes))
        
        ce_age = tf.nn.softmax_cross_entropy_with_logits_v2(logits=age_logits, labels=age_)
        ce_gender = tf.nn.softmax_cross_entropy_with_logits_v2(logits=gender_logits, labels=gender_)
        
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

    # def eval(self, x, y_age, y_gender):
    #     '''Predicts autoregressively
    #     At inference, input ys is ignored.
    #     Returns
    #     y_hat: (N, T2)
    #     '''
    #     age_logits, gender_logits, src_masks = self.encode(x, False)

    #     logging.info("Inference graph is being built. Please be patient.")
    #     pred_age = tf.argmax(age_logits, axis=1)
    #     pred_gender = tf.argmax(gender_logits, axis=1)
    #     # monitor a random sample

    #     tf.summary.text("pred_age", pred_age)
    #     tf.summary.text("pred_gender", pred_gender)
    #     summaries = tf.summary.merge_all()

    #     return pred_age, pred_gender, summaries

