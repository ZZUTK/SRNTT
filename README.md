# Super-Resolution-by-Neural-Texture-Transfer
Tensorflow implementation of the paper [Super-Resolution by Neural Texture Transfer (SRNTT)](https://arxiv.org/pdf/1804.03360.pdf).

[Project Page](http://web.eecs.utk.edu/~zzhang61/project_page/SRNTT/SRNTT.html)


## Contents
* [Pre-requisites](#Pre-requisites)
* [Dataset](#Dataset)
    * [Training set](#Training_set)
    * [Testing set](#Testing_set)
* [Easy testing](#Easy_testing)
* [Custom testing](#Custom_testing)
* [Easy training](#Easy_training)
* [Custom training](#Custom_training)
* [Acknowledgement](#Acknowledgement)
* [Contact](#Contact)

<a name="Pre-requisites">

## Pre-requisites

* TensorFlow 1.5
* Python 2.7 or Python 3.6

<a name="Dataset">

## Dataset

<a name="Training_set">

* #### Training set
This repo only provides a small training set of ten input-reference pairs for demo purpose. 
The input images and reference images are stored in `data/train/input` and `data/train/ref`, respectively.
Corresponding input and refernece images are with the same file name. 
To speed up the training process, patch matching and swapping are performed offline, 
and the swapped feature maps are saved to `data/train/map`. If you want to train your own model, please prepare your own training set or download either of the following demo training sets:

1. 11,485 input-reference pairs (size 320x320) extracted from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/). 
Each pair is extracted from the same image without overlap but considering scaling and rotation. 
```bash
$ python download_dataset.py --dataset_name DIV2K
```

2. 11,871 input-reference pairs (size 160x160) extracted from [CUFED](http://acsweb.ucsd.edu/~yuw176/event-curation.html).
Each pair is extracted from the similar images, including five degrees of similarity. 
```bash
$ python download_dataset.py --dataset_name CUFED
```

The images in training set #1 are with clearner and simpler/less texture compared to 
the training set #2.

<a name="Testing_set">

* #### Testing set
This repo includes two grounps of samples from the [CUFED5](https://drive.google.com/open?id=1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph) dataset, where each input image corresponds to five reference images with different degrees of similairty to the input image. Please download the full dataset by
```bash
$ python download_dataset.py --dataset_name CUFED5
```

<a name="Easy_testing">

## Easy Testing
```bash
$ sh test.sh

Demo of the testing process
********************
	Build Graph
********************
********************
	Load Models
********************
[*] Load SRNTT/models/SRNTT/upscale.npz SUCCESS!
[!] Load save_model/model/srntt.npz failed!
[*] Load SRNTT/models/SRNTT/srntt.npz SUCCESS!
********************
	Texture Swap
********************
>> Get VGG19 feature maps
>> Patch matching and swapping
>> Saved results to result
```
The results will be save to the folder `result`, which consists of 6 images:
* [1/6] `HR.png`, the original image.

  ![Original image](result/HR.png)

* [2/6] `LR.png`, the low-resolution (LR) image, downscaling factor 4x.

  ![LR image](result/LR.png)
  
* [3/6] `Bicubic.png`, the upscaled image by bicubic interpolation, upscaling factor 4x.

  ![Bicubic image](result/Bicubic.png)
  
* [4/6] `Ref.png`, the upscaled image by a pre-trained SR network, upscaling factor 4x.

  ![Reference image](result/Ref.png)
  
* [5/6] `Upscale.png`, the upscaled image by a pre-trained SR network, upscaling factor 4x.

  ![Upscaled image](result/Upscale.png)
  
* [6/6] `SRNTT.png`, the SR result by SRNTT, upscaling factor 4x.

  ![Upscaled image](result/SRNTT.png)

<a name="Custom_testing">

## Custom Testing
```bash
$ python main.py 
    --is_train              False 
    --input_dir             path/to/input/image/file
    --ref_dir               path/to/ref/image/file
    --result_dir            path/to/result/folder
    --ref_scale             default 1, expected_ref_scale divided by original_ref_scale
    --is_original_image     default True, whether input is original 
    --use_init_model_only   default False, whether use init model, trained with reconstruction loss only
    --use_weight_map        defualt False, whether use weighted model, trained with the weight map.
    --save_dir              path/to/a/specified/model if it exists, otherwise ignor this parameter
```
For example,
```bash
$ python main.py 
    --is_train False 
    --input_dir data/test/CUFED5/001_0.png 
    --ref_dir data/test/CUFED5/001_2.png
    --result_dir result_custom_test
    --ref_scale 1
    --is_original_image True
    --use_init_model_only True
    --use_weight_map False
    --save_dir None
```
Above is not legal command. It is just for better visualization. 
Please copy the following command and paste to the terminal.
```commandline
python main.py --is_train False --input_dir data/test/CUFED5/001_0.png --ref_dir data/test/CUFED5/001_2.png --result_dir result_custom_test --ref_scale 1 --is_original_image True --use_init_model_only True --use_weight_map False --save_dir None
```
In the folder `result_custom_test`, you will find the SRNTT result as the following, which is obtained from the init model 
that is trained only by reconstruction loss. 

![SRNTT init result](result_custom_test/SRNTT_init.png)

Please note that this repo provides three types of pre-trained SRNTT models in `SRNTT/models/SRNTT`:
* `srntt.npz` is trained by all losses, i.e., reconstruction loss, perceptual loss, texture loss, and adversarial loss.
* `srntt_init.npz` is trained by only the reconstruction loss. 
* `srntt_weighted.npz` is trained by all losses using weighted map (`--use_weight_map True`), which reduces negative effect from the reference image.

To switch between the demo models, please set `--use_init_model_only` to decide whether use `srntt_init.npz`, and set `--use_weight_map` to `False` for `srntt.npz` or `True` for `srntt_weighted.npz`.

<a name="Easy_training">

## Easy Training
To speed up the training process, we conduct patch matching and swapping first to get the swapped feature maps.
```bash
$ python offline_patchMatch_textureSwap.py

01/10
02/10
03/10
04/10
05/10
06/10
07/10
08/10
09/10
10/10
``` 
The feature maps are saved to `data/train/map`. Then, run the training script.
```bash
$ sh train.sh

Demo of the training process
********************
	Build Graph
********************
********************
	Load Models
********************
[*] Load SRNTT/models/SRNTT/upscale.npz SUCCESS!
********************
	Training
********************
Pre-train: Epoch [01/02] Batch [001/001]	ETA: 11 sec
	l_rec = 0.4702
[*] save_model/model/srntt_init.npz saved
Pre-train: Epoch [02/02] Batch [001/001]	ETA: 06 sec
	l_rec = 0.4814
[*] save_model/model/srntt_init.npz saved
Epoch [01/02] Batch [001/001]	ETA: 45 sec
	l_rec = 0.4975
	l_per = 3.5130	l_tex = 0.8971
	l_adv = 0.0000	l_dis = 5.3625
[*] save_model/model/srntt.npz saved
[*] save_model/model/discrim.npz saved
Epoch [02/02] Batch [001/001]	ETA: 26 sec
	l_rec = 0.4859
	l_per = 3.5121	l_tex = 0.8258
	l_adv = 0.0001	l_dis = -134.2220
[*] save_model/model/srntt.npz saved
[*] save_model/model/discrim.npz saved
```
The models are saved to `save_model/model`, and intermediate samples are saved to `save_model/sample`.

<a name="Custom_training">

## Custom Training
Please first prepare the input and reference images which are squared patches in the same size.
In addition, input and reference images should be stored in separated folders,
and the correspoinding input and reference images are with the same file name. Pleaser refer to the `data/train` folder for example.
Then, use `offline_patchMatch_textureSwap.py` to generate the feature maps in ahead.

```bash
$ python main.py
    --is_train True
    --save_dir folder/to/save/models
    --input_dir path/to/input/image/folder
    --ref_dir path/to/ref/image/folder
    --map_dir path/to/feature_map/folder
    --batch_size default 9
    --num_epochs default 100
    --input_size default 40, the size of LR patch, i.e., 1/4 of the HR image, set to 80 for the DIV2K dataset
    --use_weight_map defualt False, whether use the weight map that reduces negative effect 
                     from the reference image but may also decrease the sharpness.  
```
Please refer to `main.py` for more parameter settings for training.

Test on the custom training model
```bash
$ python main.py 
    --is_train              False 
    --input_dir             path/to/input/image/file
    --ref_dir               path/to/ref/image/file
    --result_dir            path/to/result/folder
    --ref_scale             default 1, expected_ref_scale divided by original_ref_scale
    --is_original_image     default True, whether input is original 
    --save_dir              the same as save_dir in training
```

<a name="Acknowledgement">

## Acknowledgement
Thanks to [Tensorlayer](https://github.com/tensorlayer/tensorlayer) for 
facilitating the implementation of this demo code. 
We have include the Tensorlayer 1.5.0 in `SRNTT/tensorlayer`.

<a name="Contact">

## Contact
[Zhifei Zhang](http://web.eecs.utk.edu/~zzhang61/)







