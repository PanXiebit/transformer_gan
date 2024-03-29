#!bin/bash


export CUDA_VISIBLE_DEVICES="4,5"
export LD_LIBRARY_PATH=/usr/local/nccl_2.3.4/lib:$LD_LIBRARY_PATH

BASE_DIR=/home/work/xiepan/xp_dial/gan_nmt/transformer_gan
# sharing embedding
python3 main_pretrain.py \
    --num_gpus 2 \
    --param_set base \
    --data_dir ${BASE_DIR}/data/en-tr/data_total/gen_data \
    --model_dir ${BASE_DIR}/data/en-tr/model_save/train_base \
    --learning_rate 2.0 \
    --batch_size  4096 \
    --max_length 50 \
    --fro src \
    --to tgt \
    --save_checkpoints_secs 1200 \
    --shared_embedding_softmax_weights true \
    --train_steps 50000 \
    --steps_between_evals 1000



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
