###
 # @Author: your name
 # @Date: 2020-05-09 15:28:53
 # @LastEditTime: 2020-05-26 18:10:40
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /ad2020/run.sh
 ###


CUDA_VISIBLE_DEVICES=1 python3 train.py  --train_features_path ../../5-8/sample_train_features.pkl  --train_labels_path ../../5-8/sample_train_labels.pkl --vocab_list ../../5-8/sample_vocab_size_list.pkl --num_heads 2 --d_model 64,16,64,32 --num_blocks 3 --d_ff 256
# CUDA_VISIBLE_DEVICES=1 python3 train.py  --train_features_path ../../5-8/train_features.pkl  --train_labels_path ../../5-8/train_labels.pkl --vocab_list ../../5-8/vocab_size_list.pkl --num_heads 8 --d_model 64,16,64,32 --num_blocks 3 --d_ff 256 --maxlen 60



# train
## files
    # --train_dense_path ../5-8/sample_train_dense_seqs.pkl\
    # --train_sparse_path ../5-8/sample_train_sparse_seqs.pkl\
    # --train_age_gender_path ../5-8/sample_train_age_gender.pkl\
# # # test
# # --test\
# # --ckpt\
# # --testdir\
# # training scheme
#     --batch_size 128\
# # --lr\
# # --warmup_steps\
# # --logdir\
#     --num_epochs 1\

# # model
#     --d_model 32\
#     --d_ff 32\
#     --num_blocks 1\
#     --num_heads 1\
#     --maxlen 50\
#     --dropout_rate\
#     # --smoothing\
#     --age_classes 2\
#     --gender_classes 10