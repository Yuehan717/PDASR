# PDASR
> Perception-Distortion Balanced ADMM Optimization for Single-Image Super-Resolution  
> Yuehan Zhang, Bo Ji, Jia Hao, and Angela Yao  
> In ECCV 2022

### Introduction
Single Image Super-Resolution (SISR) usually only does well in either objective quality or perceptual quality, due to the perception-distortion trade-off. In this paper, we proposed a two-stage model trained with low-frequency constraint and designed ADMM algorithm. Experimentally. our method achieve high perfromance in both PSNR/SSIM (objective quality) and NRQM/LPIPS (perceptual quality). Check followings for details.
> [Paper](https://arxiv.org/abs/2208.03324) | Sumpplementary Material
### Getting Start
- clone this repository  
```
git clone https://github.com/Yuehan717/PDASR  
cd PDASR/src
```
- Install dependencies. (Python >= 3.7 + CUDA)
- Require pytorch=1.9.1: [official instructions](https://pytorch.org/get-started/previous-versions/)
- Install other requirements
```
pip install -r requirements.txt
```

### Data Preparation
- Download [testing data](https://drive.google.com/drive/folders/1u7pWhYqO1Mmba76aH-_-8rUFqe0oeyW5?usp=sharing) from Google Drive
- Put data under folder or change the dir value in  
(Temporally does not support to test self-collected data)

### Testing
- Download [trained model](https://drive.google.com/drive/folders/1u7pWhYqO1Mmba76aH-_-8rUFqe0oeyW5?usp=sharing) and put it under the folder models.
- Run following command
```
python test.py --scale 4 --save test_results --templateD HAN --templateP Clique \
--dir_data [root of testing sets] --data_test Set5+Set14+B100+Urban100 \
--pre_train ../models/model_trained.pt --save_results
```
We also provide the [testing results](https://drive.google.com/drive/folders/1u7pWhYqO1Mmba76aH-_-8rUFqe0oeyW5?usp=sharing) in our paper.
### Training

Instructions coming soon  

_Our code is based on [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch). Thanks to their great work._
