# Setup Stable Diffusion Model

Since AnimateDiff is based on [Stable Diffusion](https://github.com/runwayml/stable-diffusion) v1.5, you have to download the corresponding weights in order to let the model
run properly. Here's a step-by-step guide to download all the necessary pre-trained weights for this purpose.
- Go to the [HuggingFace page](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) for Stable Diffusion v1.5;
- Open the *safety_checker* folder and download the *pytorch_model.bin* file. Move this file into our *safety_checker* folder;
- Open the *text_encoder* folder and download the *pytorch_model.bin* file. Move this file into our *text_encoder* folder;
- Open the *unet* folder and download the *diffusion_pytorch_model.bin* file. Move this file into our *unet* folder;
- Open the *vae* folder and download the *diffusion_pytorch_model.bin* file. Move this file into our *vae* folder.

