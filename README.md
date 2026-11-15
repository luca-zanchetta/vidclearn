# VidCLearn: A Continual Learning Approach for Text-to-Video Generation

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/pdf/2509.16956)

This is the official repository that contains the code implementing **VidCLearn**, our proposed continual learning approach for one-shot Text-to-Video generative models.

## Abstract
Text-to-video generation is an emerging field in generative AI, enabling the creation of realistic, semantically accurate videos from text prompts. While current models achieve impressive visual quality and alignment with input text, they typically rely on static knowledge—making it difficult to incorporate new data without retraining from scratch. To address this limitation, we propose VidCLearn, a continual learning framework for diffusion-based text-to-video generation. VidCLearn features a student-teacher architecture where the student model is incrementally updated with new text-video pairs, and the teacher model helps preserve previously learned knowledge through generative replay. Additionally, we introduce a novel temporal consistency loss to enhance motion smoothness and a video retrieval module to provide structural guidance at inference. Our architecture is also designed to be more computationally efficient than existing models retaining a satisfactory generation performance. Experimental results show VidCLearn superiority over baseline methods in terms of visual quality, semantic alignment, and temporal coherence.

## Results
**Table 1: Naive Approach: visual quality and semantic alignment.**

| Videos | FVD ↓ | FID ↓ | IS ↑ | CLIP ↑ |
|:------:|------:|------:|-----:|-------:|
| 5      | 865,824 | 9,344  | 10,272 | 32,058 |
| 10     | 935,043 | 19,915 | 9,298  | 31,002 |
| 25     | 1224,906 | 20,105 | 6,043  | 26,870 |
| 50     | 954,725 | 11,570 | 6,891  | 27,677 |
| 75     | 952,074 | 11,538 | 7,318  | 28,187 |
| 90     | 617,230 | 7,112  | 6,328  | 26,743 |

**Table 2: EWC Approach: visual quality and semantic alignment.**

| Videos | FVD ↓ | FID ↓ | IS ↑ | CLIP ↑ |
|:------:|------:|------:|-----:|-------:|
| 5  | 1096,049 | 10,155 | 10,820 | 31,839 |
| 10 | 1210,202 | 25,408 | 9,589  | 30,278 |
| 25 | 1593,644 | 18,167 | 6,751  | 28,508 |
| 50 | 1264,515 | 12,656 | 6,942  | 28,200 |
| 75 | 844,169  | 7,481  | 7,091  | 25,641 |
| 90 | 901,032  | 6,162  | 8,003  | 28,288 |

**Table 3: VidCLearn Approach: visual quality and semantic alignment.**

| Videos | FVD ↓ | FID ↓ | IS ↑ | CLIP ↑ |
|:------:|------:|------:|-----:|-------:|
| 5  | 740,210 | 8,505  | 12,002 | 32,800 |
| 10 | 820,430 | 15,704 | 10,885 | 31,754 |
| 25 | 980,112 | 14,002 | 9,123  | 29,932 |
| 50 | 852,764 | 10,200 | 9,005  | 30,125 |
| 75 | 765,821 | 6,512  | 10,304 | 29,342 |
| 90 | **734,320** | **3,155**  | **12,832** | **32,903** |


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
Luca Zanchetta [[Email](zanchetta.1848878@studenti.uniroma1.it)]

Lorenzo Papa [[Email](lorenzo.papa@esa.int)] [[Scholar](https://scholar.google.com/citations?user=P64hj-4AAAAJ&hl=it)]

Luca Maiano [[Email](maiano@diag.uniroma1.it)] [[Scholar](https://scholar.google.com/citations?user=FZyBVqkAAAAJ&hl=en)]

Irene Amerini [[Email](amerini@diag.uniroma1.it)] [[Scholar](https://scholar.google.com/citations?user=4ZDhr6UAAAAJ&hl=en)]

## Acknowledgements
The code is based on [Tune-A-Video](https://github.com/showlab/Tune-A-Video) and [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4). The data has been taken from the [DAVIS Dataset](https://davischallenge.org/davis2017/code.html) and has been manipulated accordingly.

## Cite Us
[https://doi.org/10.48550/arXiv.2509.16956](https://doi.org/10.48550/arXiv.2509.16956)
