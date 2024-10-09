# VidCLearn: Continual Learning for Text-to-Video Generation

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)]()

This is the official repository that contains the code implementing **VidCLearn**, our proposed continual learning approach for one-shot Text-to-Video generative models. **Notice that this work is still under development!**

## Abstract
*Work in progress...*

## Results
*Work in progress...*

## Setup
- Download and Install the [Docker Engine](https://www.docker.com/products/docker-desktop/);
- Make sure the Docker Engine has been started;
- Open a terminal;
- Pull our Docker image by launching the following command:
  ```
  $ docker pull lucazanchetta/vidclearn:latest
  ```
- Run a Docker container with the pulled image (a GPU is required):
  ```
  $ docker run --gpus='"device=<device_number>"' --name <container_name> -it lucazanchetta/vidclearn:latest
  ```

## How to run the code

### Training
- Make sure to have at least 37000 MiB VRAM free for this process to be executed (with default configuration settings);
- Ensure the configuration settings are properly tailored to your needs before proceeding:
  ```
  $ nano configs/training.yaml
  ```
- Run the following command and wait for the completion of the training process:
  ```
  $ accelerate launch -m scripts.train
  ```

### Inference
- Make sure to have at least 8000 MiB VRAM free for this process to be executed (with default configuration settings);
- Ensure the configuration settings are properly tailored to your needs before proceeding:
  ```
  $ nano configs/inference.yaml
  ```
- Run the following command and wait for the completion of the inference process:
  ```
  $ accelerate launch -m scripts.inference
  ```
- Enjoy the generated video on `inference_samples` folder.

### Evaluation
- Make sure to have at least 3000 MiB VRAM free for this process to be executed (with default configuration settings);
- Ensure the configuration settings are properly tailored to your needs before proceeding:
  ```
  $ nano configs/evaluation.yaml
  ```
- Run the following command and wait for the completion of the evaluation process:
  ```
  $ accelerate launch -m scripts.evaluation
  ```
- Enjoy the results!

## Authors
Luca Zanchetta [[Email]()] [[GitHub]()] [[Scholar]()]

Lorenzo Papa [[Email]()] [[GitHub]()] [[Scholar]()]

Luca Maiano [[Email]()] [[GitHub]()] [[Scholar]()]

Irene Amerini [[Email]()] [[GitHub]()] [[Scholar]()]

## Acknowledgements
The code is based on [Tune-A-Video](https://github.com/showlab/Tune-A-Video) and [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4). The data has been taken from the [DAVIS Dataset](https://davischallenge.org/davis2017/code.html) and has been manipulated accordingly.

## Cite Us
*Work in progress...*
