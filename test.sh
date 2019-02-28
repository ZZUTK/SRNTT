#!/usr/bin/env bash
echo Demo of the testing process

# test on default SRNTT model
python main.py \
    --is_train False \
    --input_dir data/test/CUFED5/001_0.png \
    --ref_dir data/test/CUFED5/001_2.png \
    --use_init_model_only False \
    --result_dir demo_testing_srntt

## test on default SRNTT-l2 model
#python main.py \
#    --is_train False \
#    --input_dir data/test/CUFED5/001_0.png \
#    --ref_dir data/test/CUFED5/001_2.png \
#    --use_init_model_only True \
#    --result_dir demo_testing_srntt-l2

## test on your own model
#python main.py \
#    --is_train False \
#    --input_dir data/test/CUFED5/001_0.png \
#    --ref_dir data/test/CUFED5/001_2.png \
#    --use_init_model_only False \
#    --result_dir demo_testing_yours \
#    --save_dir your_model_dir