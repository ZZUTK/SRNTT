#!/usr/bin/env bash
echo Demo of the training process
python main.py \
    --is_train True \
    --use_pretrained_model False \
    --num_init_epochs 2 \
    --num_epochs 2

