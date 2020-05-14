###
 # @Author: your name
 # @Date: 2020-05-09 15:28:53
 # @LastEditTime: 2020-05-14 15:21:04
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /ad2020/run.sh
 ###


CUDA_VISIBLE_DEVICES=1 python3 train.py \
--train_features_path ../../5-8/sample_train_features.pkl \
--train_labels_path ../../5-8/sample_train_labels.pkl \
--pretrained_emb_path ../../pre_embedding/emb \
--vocab_list ../../5-8/sample_vocab_size_list.pkl \
--num_heads 2 \
--d_model 64 \
--num_blocks 3 \
--d_ff 512 \
--target_label gender \
--pretrain 0 \
--logdir 2_64_3_512_gender_0



# gender训练
CUDA_VISIBLE_DEVICES=1 python3 train.py \
--train_features_path ../../5-8/train_features.pkl \
--train_labels_path ../../5-8/train_labels.pkl \
--eval_features_path ../../5-8/eval_features.pkl \
--eval_labels_path ../../5-8/eval_labels.pkl \
--pretrained_emb_path ../../pre_embedding/emb \
--vocab_list ../../5-8/vocab_size_list.pkl \
--num_heads 2 \
--d_model 64 \
--num_blocks 3 \
--d_ff 512 \
--target_label gender \
--pretrain 1 \
--trainable 1 \
--logdir 2_64_3_512_gender_1_0


# age训练
CUDA_VISIBLE_DEVICES=1 python3 train.py \
--train_features_path ../../5-8/train_features.pkl \
--train_labels_path ../../5-8/train_labels.pkl \
--pretrained_emb_path ../../pre_embedding/emb \
--vocab_list ../../5-8/vocab_size_list.pkl \
--num_heads 2 \
--d_model 64 \
--num_blocks 3 \
--d_ff 512 \
--target_label age \
--pretrain 1 \
--logdir 2_64_3_512_age_1



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