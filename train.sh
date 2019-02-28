#!/usr/bin/env bash
echo Demo of the training process

training_set=${1-CUFED}

# # download training set
python download_dataset.py --dataset_name ${training_set}

# # calculate the swapped feature map in the offline manner
python offline_patchMatch_textureSwap.py --data_folder data/train/${training_set}


# # train a new model
python main.py \
    --is_train True \
    --input_dir data/train/${training_set}/input \
    --ref_dir data/train/${training_set}/ref \
    --map_dir data/train/${training_set}/map_321 \
    --use_pretrained_model False \
    --num_init_epochs 2 \
    --num_epochs 2 \
    --save_dir demo_training_srntt

## train based on a pre-trained model
#python main.py \
#    --is_train True \
#    --input_dir data/train/${training_set}/input \
#    --ref_dir data/train/${training_set}/ref \
#    --map_dir data/train/${training_set}/map_321 \
#    --use_pretrained_model True \
#    --num_init_epochs 0 \
#    --num_epochs 2 \
#    --save_dir demo_training_srntt

