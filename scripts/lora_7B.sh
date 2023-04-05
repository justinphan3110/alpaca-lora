#!/bin/bash

source /opt/rh/devtoolset-10/enable

srun --nodes=1 --gpus-per-node=4 WORLD_SIZE=4 deepspeed finetune.py \
    --base_model='/data/long_phan/llama_hf_weights/llama-7b' \
    --data_path='yahma/alpaca-cleaned' \
    --output_dir='out/lora-alpaca_7B' \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --batch_size=1024 \
    --micro_batch_size=128