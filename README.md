# AdCo
<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/AdCo-v2.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
   <img src="https://img.shields.io/badge/licence-GNU-green">
</a>   

AdCo is a contrastive-learning based self-supervised learning methods. 

Copyright (C) 2020 Qianjiang*, Xiao Wang*, Wei Hu, Guo-Jun Qi

License: MIT for academic use.

Contact: Guo-Jun Qi (guojunq@gmail.com)

## Introduction
Contrastive learning relies on constructing a collection of negative examples that are sufficiently hard to discriminate against positive queries when their representations are self-trained. Existing contrastive learning methods either maintain a queue of negative samples over minibatches while only a small portion of them are updated in an iteration, or only use the other examples from the current minibatch as negatives. They could not closely track the change of the learned representation over iterations by updating the entire queue as a whole, or discard the useful information from the past minibatches. Alternatively, we present to directly learn a set of negative adversaries playing against the self-trained representation. Two players, the representation network and negative adversaries, are alternately updated to obtain the most challenging negative examples against which the representation of positive queries will be trained to discriminate. We further show that the negative adversaries are updated towards a weighted combination of positive queries by maximizing the adversarial contrastive loss, thereby allowing them to closely track the change of representations over time. Experiment results demonstrate the proposed Adversarial Contrastive (AdCo) model not only achieves superior performances (a top-1 accuracy of 73.2% over 200 epochs and 75.7% over 800 epochs with linear evaluation on ImageNet), but also can be pre-trained more efficiently with much shorter GPU time and fewer epochs.


## Installation  
AdCo requires single machine with 8*V100 GPUs, CUDA version 10.1 or higher. 
### 1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
### 2. Clone the repository in your computer 
```
git clone git@github.com:maple-research-lab/AdCo.git && cd AdCo
```

### 3. Build dependencies.   
You have two options to install dependency on your computer:
#### 3.1 Install with pip and python(Ver 3.6.9).
##### 3.1.1[`install pip`](https://pip.pypa.io/en/stable/installing/).
##### 3.1.2  Install dependency in command line.
```
pip install -r requirements.txt --user
```
If you encounter any errors, you can install each library one by one:
```
pip install torch==1.7.1
pip install torchvision==0.8.2
pip install numpy==1.19.5
pip install Pillow==5.1.0
pip install tensorboard==1.14.0
pip install tensorboardX==1.7
```

#### 3.2 Install with anaconda
##### 3.2.1 [`install conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html). 
##### 3.2.2 Install dependency in command line
```
conda create -n AdCo python=3.6.9
conda activate AdCo
pip install -r requirements.txt 
```
Each time when you want to run my code, simply activate the environment by
```
conda activate AdCo
conda deactivate(If you want to exit) 
```


## Usage

### Unsupervised Training
This implementation only supports multi-gpu, DistributedDataParallel training, which is faster and simpler; single-gpu or DataParallel training is not supported. Before training, please download [ImageNet2012 Dataset](http://image-net.org/challenges/LSVRC/2012/) to "./datasets/imagenet2012".
#### Single Crop
##### 1 Without symmetrical loss:
```
python3 main_adco.py --data=./datasets/imagenet2012 --dist_url=tcp://localhost:10001 --sym=0
```
##### 2 With symmetrical loss:
```
python3 main_adco.py --data=./datasets/imagenet2012 --dist_url=tcp://localhost:10001 --sym=1
```
#### Multi Crop
```
python3 main_adco.py --data=./datasets/imagenet2012 --dist_url=tcp://localhost:10001 --multi_crop=1
```
So far we don't support multi crop with symmetrical loss. 

### Linear Classification
With a pre-trained model, we can easily evaluate its performance on ImageNet with:
```
python3 lincls.py --data=./datasets/imagenet2012 --dist-url=tcp://localhost:10001 --pretrained=input.pth.tar
```
Performance:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">pre-train<br/>network</th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">Crop</th>
<th valign="bottom">Symmetrical<br/>Loss</th>
<th valign="bottom">AdCo<br/>top-1 acc.</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">Single</td>
<td align="center">No</td>
<td align="center">68.6</td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">Multi</td>
<td align="center">No</td>
<td align="center">73.2</td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">800</td>
<td align="center">Single</td>
<td align="center">No</td>
<td align="center">72.8</td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">800</td>
<td align="center">Multi</td>
<td align="center">No</td>
<td align="center">75.7</td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">Single</td>
<td align="center">Yes</td>
<td align="center">70.6</td>
</tr>
</tbody></table>

### Transfering to VOC07 Classification
#### 1 Download [Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and extract to ./datasets/voc
#### 2 Linear Evaluation:
```
# in VOC_CLF folder
python3 main.py --data=../datasets/voc --pretrained=../input.pth.tar
```
Here VOC directory should be the directory includes "vockit" directory.

### Transfer to Places205 Classification
#### 1 Download [Dataset](http://places.csail.mit.edu/user/index.php) and extract to ./datasets/places205
#### 2 Linear Evaluation:
```
python3 lincls.py --dataset=Place205 --sgdr=1 --data=./datasets/places205 --lr=5 --dist-url=tcp://localhost:10001 --pretrained=input.pth.tar
```

### Transfer to Object Detection
Modified from [MoCo Detection](https://github.com/facebookresearch/moco/tree/master/detection)
1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

1. Convert a pre-trained AdCo model to detectron2's format:
   ```
   # in detection folder
   python3 convert-pretrain-to-detectron2.py input.pth.tar output.pkl
   ```

1. download [VOC Dataset](http://places.csail.mit.edu/user/index.php) and [COCO Dataset](https://cocodataset.org/#download) under "./detection/datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

1. Run training:
   ```
   # in detection folder
   python train_net.py --config-file configs/pascal_voc_R_50_C4_24k_adco.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./output.pkl
   ```


## Citation:
[AdCo: Adversarial Contrast for Efficient Learning of Unsupervised Representations from Self-Trained Negative Adversaries](https://arxiv.org/pdf/2011.08435.pdf)
```
@article{hu2020adco,
  title={AdCo: Adversarial Contrast for Efficient Learning of Unsupervised Representations from Self-Trained Negative Adversaries},
  author={Hu, Qianjiang and Wang, Xiao and Hu, Wei and Qi, Guo-Jun},
  journal={arXiv preprint arXiv:2011.08435},
  year={2020}
}
```

