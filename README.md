# AdCo
<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/AdCo-v1.0.0-green">
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
CUDA version should be 10.1 or higher. 
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
#### 4 Prepare the ImageNet dataset
##### 4.1 Download the [ImageNet2012 Dataset](http://image-net.org/challenges/LSVRC/2012/) under "./datasets/imagenet2012".
##### 4.2 Go to path "./datasets/imagenet2012/val"
##### 4.3 move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Usage

### Unsupervised Training
This implementation only supports multi-gpu, DistributedDataParallel training, which is faster and simpler; single-gpu or DataParallel training is not supported.
#### Single Crop
##### 1 Without symmetrical loss:
```
python3 main_adco.py --sym=0 --lr=0.03 --memory_lr=3 --moco_t=0.12 --mem_t=0.02 --data=./datasets/imagenet2012 --dist_url=tcp://localhost:10001 
```
##### 2 With symmetrical loss:
```
python3 main_adco.py --sym=1 --lr=0.03 --memory_lr=3 --moco_t=0.12 --mem_t=0.02 --data=./datasets/imagenet2012 --dist_url=tcp://localhost:10001
```
#### 3 setting different numbers of negative samples:
```
# e.g., training with 8192 negative samples and symmetrical loss
python3 main_adco.py --sym=1 --lr=0.04 --memory_lr=3 --moco_t=0.14 --mem_t=0.03 --cluster 8192 --data=./datasets/imagenet2012 --dist_url=tcp://localhost:10001
```

#### Multi Crop
```
python3 main_adco.py --multi_crop=1 --lr=0.03 --memory_lr=3 --moco_t=0.12 --mem_t=0.02 --data=./datasets/imagenet2012 --dist_url=tcp://localhost:10001
```



<font size=0 >So far we have yet to support multi crop with symmetrical loss, the feature will be added in future.</font>

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
<th valign="bottom">Model<br/>Link</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">Single</td>
<td align="center">No</td>
<td align="center">68.6</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EUZnXZAGrDFAoHEy7HxYsJgBqk7VDOjIGa1wUWXk2FArbQ?e=Gs9rXD">model</a></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">Multi</td>
<td align="center">No</td>
<td align="center">73.2</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EQKYrTt0nolMrKYLQ-FPHR4Be6ZA-pPXa9HQArhFQqEr2g?e=A2gCdH">model</a></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">800</td>
<td align="center">Single</td>
<td align="center">No</td>
<td align="center">72.8</td>
<td align="center"><a href="">model</a></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">800</td>
<td align="center">Multi</td>
<td align="center">No</td>
<td align="center">75.7</td>
<td align="center">None</td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">Single</td>
<td align="center">Yes</td>
<td align="center">70.6</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EQAk2hTJo3NPl8TggXdzE6wB4yGEMD8_pRVcRhxlYCpSLQ?e=8wdc4a">model</a></td>
</tr>
</tbody></table>

Really sorry that we can't provide multi-800 model, which is because that we train it with 32 internal GPUs and we can't download it because of company regulations. For downstream tasks, we found single-800 also had similar performance. Thus, we suggested you to use this [model]() for downstream purposes.


Performance with different negative samples:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">pre-train<br/>network</th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">negative<br/>samples </th>
<th valign="bottom">Symmetrical<br/>Loss</th>
<th valign="bottom">AdCo<br/>top-1 acc.</th>
<th valign="bottom">Model<br/>Link</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">65536</td>
<td align="center">No</td>
<td align="center">68.6</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EUZnXZAGrDFAoHEy7HxYsJgBqk7VDOjIGa1wUWXk2FArbQ?e=Gs9rXD">model</a></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">65536</td>
<td align="center">Yes</td>
<td align="center">70.6</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EQAk2hTJo3NPl8TggXdzE6wB4yGEMD8_pRVcRhxlYCpSLQ?e=8wdc4a">model</a></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">16384</td>
<td align="center">No</td>
<td align="center">68.6</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/ESxqq4V9MtVHmo_u4uotVQ0BBpLy8RuCSCILrIqsqXN_6g?e=p2HAXH">model</a></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">16384</td>
<td align="center">Yes</td>
<td align="center">70.2</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EamboKxWLFlOr6qJaQUWWDIB7ut_zAituINY9PT69fYhFQ?e=YLEEde">model</a></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">8192</td>
<td align="center">No</td>
<td align="center">68.4</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EcWjd4-4tepHsqp-Idd81lcBf61T2CyQ3sc1koEqMm74xg?e=BWqBYx">model</a></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">8192</td>
<td align="center">Yes</td>
<td align="center">70.2</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/ERxg2B8-rihKs3Wm78cT76EB9euFLDWlHkvyMAAjJBODOQ?e=l55cMa">model</a></td>	
</tr>
</tbody></table>


The performance is obtained on a single machine with 8*V100 GPUs.


### Transfering to VOC07 Classification
#### 1 Download [Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) under "./datasets/voc"
#### 2 Linear Evaluation:
```
cd VOC_CLF
python3 main.py --data=../datasets/voc --pretrained=../input.pth.tar
```
Here VOC directory should be the directory includes "vockit" directory.

### Transfer to Places205 Classification
#### 1 Download [Dataset](http://places.csail.mit.edu/user/index.php) under "./datasets/places205"
#### 2 Linear Evaluation:
```
python3 lincls.py --dataset=Place205 --sgdr=1 --data=./datasets/places205 --lr=5 --dist-url=tcp://localhost:10001 --pretrained=input.pth.tar
```

### Transfer to Object Detection
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
   cd detection
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

