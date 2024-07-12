# lightweight-text-to-video

The code is based on [AnimateDiff](https://github.com/guoyww/AnimateDiff).

## Setup

The first thing you can do is cloning our repository, by launching the following command:
```
$ git clone https://github.com/luca-zanchetta/lightweight-text-to-video/tree/main
```
Then, launch the following command:
```
$ cd lightweight-text-to-video
```

### Download Panda Dataset
Once you have cloned our repository, you need to download the [Panda-70M](https://snap-research.github.io/Panda-70M/) dataset. You can find a detailed guide [here](https://github.com/luca-zanchetta/lightweight-text-to-video/blob/main/animatediff/data/README.md). 

### Download Pre-Trained Stable Diffusion
After having downloaded the dataset, the next step is to download pre-trained weights of the [Stable Diffusion model](https://arxiv.org/abs/2112.10752). You can find a detailed guide [here](https://github.com/luca-zanchetta/lightweight-text-to-video/blob/main/models/StableDiffusion/README.md).

## Training
The training script supports the execution on a distributed environment. We recommend to properly configure the configuration files [image_finetune.yaml](https://github.com/luca-zanchetta/lightweight-text-to-video/blob/main/configs/training/v1/image_finetune.yaml) and [training.yaml](https://github.com/luca-zanchetta/lightweight-text-to-video/blob/main/configs/training/v1/training.yaml) in advance, in order to reach the desired behavior. Moreover, we recommend the usage of a GPU consisting of at least 40GB VRAM (e.g., DGX A100 GPU). **Important remark**: It is possible to execute the training script only after having downloaded the dataset and the pre-trained Stable Diffusion weights.

For properly training our model, open a terminal in the current directory and launch the following commands:
```
$ torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/v1/image_finetune.yaml
```
for running the U-Net finetuning, and
```
$ torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/v1/training.yaml
```
for running the motion module training. Please execute these commands in this order. You can obviously change the values for *--nnodes* and *--nproc_per_node* according to your specific needs and capabilities. You will find checkpoints in *outputs/image_finetune/checkpoints* and *outputs/training/checkpoints* folders, respectively.

## Inference
The inference script supports the execution on a distributed environment. We recommend to properly configure the configuration file [inference.yaml](https://github.com/luca-zanchetta/lightweight-text-to-video/blob/main/configs/inference/inference.yaml) in advance, in order to reach the desired behavior. Moreover, we recommend the usage of a GPU consisting of at least
10GB VRAM (e.g., the one provided by Colab free). **Important remark**: It is possible to execute the inference script only after having executed the training script with both commands.

For properly performing inference with our model, open a terminal in the current directory and launch the following command:
```
$ torchrun --nnodes=1 --nproc_per_node=1 inference.py --config configs/inference/inference.yaml
```
You can obviously change the values for *--nnodes* and *--nproc_per_node* according to your specific needs and capabilities. You will find results in *inference_samples* folder.

## Evaluation
*Work in progress...*
