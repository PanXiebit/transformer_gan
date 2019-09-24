#!bin/bash
# train from scratch without pre-train.

export CUDA_VISIBLE_DEVICES="7"
export LD_LIBRARY_PATH=/usr/local/nccl_2.3.4/lib:$LD_LIBRARY_PATH

# sharing embedding
BASE_DIR=/home/work/xiepan/xp_dial/gan_nmt/transformer_gan
# sharing embedding
python3 main_sagan.py \
    --num_gpus 1 \
    --param_set base \
    --data_dir  ${BASE_DIR}/data/en-tr/data_total/gen_data   \
    --model_dir ${BASE_DIR}/data/en-tr/data_total/model_save/train_gan_4 \
    --learning_rate 2 \
    --batch_size 150 \
    --max_length 50 \
    --fro src \
    --to tgt \
    --train_steps 20000 \
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
