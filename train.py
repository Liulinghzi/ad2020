'''
@Author: your name
@Date: 2019-09-23 18:54:24
@LastEditTime: 2020-05-14 15:19:24
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /transformer-master/train.py
'''
# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import tensorflow as tf

from model import Transformer
from tqdm import tqdm
from data_load import get_batch
from utils import save_hparams, save_variable_specs, calc_metric
import os
from hparams import Hparams
import math
import time
import logging

logging.basicConfig(level=logging.INFO)


logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)


logging.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batch(hp.train_features_path, hp.train_labels_path, hp.target_label, hp.maxlen, hp.batch_size, shuffle=True)
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval_features_path, hp.eval_labels_path, hp.target_label, hp.maxlen , hp.batch_size, shuffle=False)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
sparse_features, dense_features, labels = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load model")
m = Transformer(hp)
loss, train_op, global_step, train_summaries = m.train(sparse_features, dense_features, labels, hp.target_label)
pred_target, target, eval_summaries = m.eval(sparse_features, dense_features, labels, hp.target_label)
# pred_age, pred_gender,pred_age_gender = m.infer(sparse_features, dense_features)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)


with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)

    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)

        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss) # train loss
            print(_loss)

            logging.info("# test evaluation")
            _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
            summary_writer.add_summary(_eval_summaries, _gs)

            acc = calc_metric(sess, pred_target, target, num_eval_batches, num_eval_samples)
            print('eval acc: ', acc)

            logging.info("# write results")
            model_output = "ckpt_%02d" % (epoch)
            # if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)
            # translation = os.path.join(hp.evaldir, model_output)
            # with open(translation, 'w') as fout:
            #     fout.write("\n".join(hypotheses))

            logging.info("# save models")
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)
    summary_writer.close()


logging.info("Done")
