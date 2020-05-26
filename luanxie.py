'''
@Author: your name
@Date: 2020-05-26 17:48:12
@LastEditTime: 2020-05-26 17:49:50
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ad2020/luanxie.py
'''
import tensorflow as tf

sess = tf.Session()

a = tf.constant([
    list(range(10)),list(range(10)),list(range(10)),list(range(10)),list(range(10)),list(range(10)),list(range(10)),list(range(10)),
    ])
print(sess.run(a))