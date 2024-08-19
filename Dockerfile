# Use the official PyTorch image from Docker Hub
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install nano package for live editing & update certificates
RUN apt-get update
RUN apt-get install nano

# Install required parts of OpenGL library
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0

# Install additional Python dependencies
RUN pip install -r requirements.txt