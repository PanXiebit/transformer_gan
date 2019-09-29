#!bin/bash


export CUDA_VISIBLE_DEVICES="1,2"
export LD_LIBRARY_PATH=/usr/local/nccl_2.3.4/lib:$LD_LIBRARY_PATH

BASE_DIR=/home/work/xiepan/xp_dial/gan_nmt/transformer_gan
ROLL_LEN=1

# sharing embedding
python3 main_pre_gan_2.py \
    --num_gpus 2 \
    --roll_num 5 \
    --param_set base \
    --data_dir ${BASE_DIR}/data/en-tr/data_total/gen_data \
    --model_dir ${BASE_DIR}/data/en-tr/model_save/train_base_gan_3 \
    --pretrain_dir ${BASE_DIR}/data/en-tr/model_save/train_base \
    --learning_rate 2.0 \
    --batch_size  800 \
    --max_length 50 \
    --roll_len ${ROLL_LEN} \
    --fro src \
    --to tgt \
    --save_checkpoints_secs 1200 \
    --train_steps 5000 \
    --steps_between_evals 200 \
    --extra_decode_length 0 \
    --shared_embedding_softmax_weights true


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
