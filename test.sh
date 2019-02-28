#!/usr/bin/env bash
echo Demo of the testing process
python main.py \
    --is_train False \
    --input_dir data/test/CUFED5/001_0.png \
    --ref_dir data/test/CUFED5/001_2.png \
    --use_init_model_only False \
    --result_dir demo_testing

python main.py \
    --is_train False \
    --input_dir data/test/CUFED5/001_0.png \
    --ref_dir data/test/CUFED5/001_2.png \
    --use_init_model_only True \
    --result_dir demo_testing
