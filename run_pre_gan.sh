#!bin/bash


export CUDA_VISIBLE_DEVICES="6,7"
export LD_LIBRARY_PATH=/usr/local/nccl_2.3.4/lib:$LD_LIBRARY_PATH

BASE_DIR=/home/work/xiepan/xp_dial/gan_nmt/transformer_sagan
BASE_DIR2=/home/work/xiepan/xp_dial/gan_nmt/transformer_gan


# sharing embedding
python3 main_pre_gan.py \
    --num_gpus 2 \
    --roll_num 1 \
    --param_set base \
    --data_dir ${BASE_DIR}/data/en-tr/data_total/gen_data \
    --model_dir ${BASE_DIR2}/data/en-tr/model_save/train_base_gan \
    --pretrain_dir /home/work/xiepan/xp_dial/gan_nmt/transformer_gan/data/en-tr/model_save/train_base \
    --learning_rate 2.0 \
    --batch_size  3000 \
    --max_length 50 \
    --fro src \
    --to tgt \
    --save_checkpoints_secs 1200 \
    --train_steps 35000 \
    --steps_between_evals 1000 \
    --extra_decode_length 20



# not sharing embedding
#python3 transformer_pretrain_main.py \
#    --num_gpus 1 \
#    --param_set base \
#    --data_dir /home/work/xiepan/xp_dial/gan_nmt/transformer_rl/data/en-tr/v0/gen_data \
#    --model_dir /home/work/xiepan/xp_dial/gan_nmt/transformer_rl/model_save/en-tr/unshare/train \
#    --learning_rate 2.0 \
#    --batch_size  10000 \
#    --max_length 50 \
#    --fro src \
#    --to tgt \
#    --save_checkpoints_secs 1200 \
#    --train_steps 40000 \
#    --steps_between_evals 2000
