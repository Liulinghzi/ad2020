###
 # @Author: your name
 # @Date: 2020-05-09 15:28:53
 # @LastEditTime: 2020-05-09 18:18:06
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /ad2020/run.sh
 ###


python3 train.py \

# train
## files
    --train_dense_path ../5-8/sample_train_dense_seqs.pkl\
    --train_sparse_path ../5-8/sample_train_sparse_seqs.pkl\
    --train_age_gender_path ../5-8/sample_train_age_gender.pkl\
# # test
# --test\
# --ckpt\
# --testdir\
# training scheme
    --batch_size 128\
# --lr\
# --warmup_steps\
# --logdir\
    --num_epochs 1\

# model
    --d_model 32\
    --d_ff 32\
    --num_blocks 1\
    --num_heads 1\
    --maxlen 50\
    --dropout_rate\
    # --smoothing\
    --age_classes 2\
    --gender_classes 10