import os
import shutil

# Lists
train_list = []
test_list = []

# Source and destination folder paths
source_folder = './animatediff/data/MSRVTT/videos/all'
train_folder = './animatediff/data/MSRVTT/videos/train'
test_folder = './animatediff/data/MSRVTT/videos/test'

# Create train list
with open('./animatediff/data/MSRVTT/videos/train_list_new.txt', 'r') as file:
    content = file.read()
    train_list = content[:-1].split('\n')

# Create test list
with open('./animatediff/data/MSRVTT/videos/test_list_new.txt', 'r') as file:
    content = file.read()
    test_list = content[:-1].split('\n')

# Copy elements
for filename in os.listdir(source_folder):
    if filename[:-4] in train_list:
        source_file_path = os.path.join(source_folder, filename)
        shutil.copy(source_file_path, train_folder)
        print(f"[INFO] Copied {filename} into TRAIN")
    elif filename[:-4] in test_list:
        source_file_path = os.path.join(source_folder, filename)
        shutil.copy(source_file_path, test_folder)
        print(f"[INFO] Copied {filename} into TEST")