###
 # @Author: your name
 # @Date: 2020-05-09 15:28:53
 # @LastEditTime: 2020-05-12 20:18:58
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /ad2020/run.sh
 ###


python3 test.py  --test_features_path ../5-8/test_features.pkl  --test_labels_path ../5-8/test_labels.pkl --vocab_list ../5-8/vocab_size_list.pkl --num_heads 2 --d_model 4  --ckpt log/1
