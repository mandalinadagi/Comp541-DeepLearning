# EDSR- Julia
Repository for EDSR model[1] implementation by using julia and Knet. 

1. Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and
Kyoung Mu Lee. Enhanced deep residual networks for single
image super-resolution. In CVPRW, 2017
https://arxiv.org/pdf/1707.02921.pdf

```
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}
```
## Train
To start training, simply run ```julia train.jl```
```
usage: train.jl [--scale SCALE] [--res_scale RES_SCALE]
                [--model_type MODEL_TYPE] [--res_block RES_BLOCK]
                [--output_features OUTPUT_FEATURES] [--data DATA]
                [--batchsize BATCHSIZE] [--patchsize PATCHSIZE]
                [--lr LR] [--decay DECAY] [--epochs EPOCHS]
                [--evaluate_every EVALUATE_EVERY]
                [--decay_no DECAY_NO] [--output_dir OUTPUT_DIR]
                [--result_dir RESULT_DIR] [-h]

EDSR Implementation in Knet

optional arguments:
  --scale SCALE         super-resolution scale (type: Int64, default:
                        2)
  --res_scale RES_SCALE
                        residual scaling (type: Float64, default: 1.0)
  --model_type MODEL_TYPE
                        type of model one of:
                        [baseline(16ResBlocks,64outputfeature),
                        original(32ResBlocks,256outputfeature)]
                        (default: "baseline")
  --res_block RES_BLOCK
                        number of residual blocks (type: Int64,
                        default: 16)
  --output_features OUTPUT_FEATURES
                        number of output feature channels (type:
                        Int64, default: 64)
  --data DATA           dataset directory (default:
                        "/kuacc/users/ckorkmaz14/comp541_project/data/")
  --batchsize BATCHSIZE
                        input batch size for training (type: Int64,
                        default: 16)
  --patchsize PATCHSIZE
                        input patch size for training (type: Int64,
                        default: 48)
  --lr LR               learning rate (type: Float64, default: 0.0001)
  --decay DECAY         learning rate decay (type: Float64, default:
                        0.5)
  --epochs EPOCHS       number of training epochs (type: Int64,
                        default: 300)
  --evaluate_every EVALUATE_EVERY
                        report PSNR in n iterations (type: Int64,
                        default: 10)
  --decay_no DECAY_NO   decay learning rate after nth epoch (type:
                        Int64, default: 200)
  --output_dir OUTPUT_DIR
                        output directory for saving model (default:
                        "/kuacc/users/ckorkmaz14/comp541_project/edsr/models")
  --result_dir RESULT_DIR
                        output directory for saving model (default:
                        "/kuacc/users/ckorkmaz14/comp541_project/edsr/results")
  -h, --help            show this help message and exit
```
## Test
You can find the result images from ```../edsr/results``` folder and models from ```../edsr/models```

| Model | Scale | File name (.jld2) | Parameters | **PSNR** | Loss |
|  ---  |  ---  | ---       | ---        | ---  | --- |
| **EDSR** | 2 | edsr_scale2_baseline_model | 1.37 M | 33.48 dB | 0.014 |
| | 3 | edsr_scale3_baseline_model | 1.55 M | 29.93 dB | 0.020 |
| | 4 | edsr_scale4_baseline_model | 1.52 M | 28.04 dB | 0.025 | 


## Results for EDSR Baseline Model 
## **Scale 2:**
![super-resolution image scale2](https://github.com/mandalinadagi/Comp541-DeepLearning/blob/master/results/result_scale2.png)
## **Scale 3:**
![super-resolution image scale3](https://github.com/mandalinadagi/Comp541-DeepLearning/blob/master/results/result_scale3.png)
## **Scale 4:**
![super-resolution image scale4](https://github.com/mandalinadagi/Comp541-DeepLearning/blob/master/results/results_scale4.png)
