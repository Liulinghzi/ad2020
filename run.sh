###
 # @Author: your name
 # @Date: 2020-05-09 15:28:53
 # @LastEditTime: 2020-05-09 16:18:03
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /ad2020/run.sh
 ###


python3 train.py \

# train
## files
--train\
--train_age_gender\
# test
--test\
--ckpt\
--testdir\
# training scheme
--batch_size 128\
--lr\
--warmup_steps\
--logdir\
--num_epochs\

# model
--d_model\
--d_ff\
--num_blocks\
--num_heads\
--maxlen 50\
--dropout_rate\
--smoothing\
--age_classes\
--gender_classes\
--age_classes\