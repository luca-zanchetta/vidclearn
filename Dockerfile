# Use the official PyTorch image from Docker Hub
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install nano package for live editing
RUN apt-get update
RUN apt-get install nano

# Install tmux for keeping terminal sessions alive
RUN apt-get install -y tmux

# Install any additional Python dependencies if needed
RUN pip install -r requirements.txt
RUN pip install "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Specify the command to run when the container starts
# CMD ["torchrun", "--nnodes=1", "--nproc_per_node=1", "train.py", "--config", "configs/training/v1/training.yaml"]