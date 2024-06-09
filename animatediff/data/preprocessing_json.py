import json

# Lists
train_list = []
test_list = []
train_captions = []
test_captions = []

# Source folder paths
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
    
    
with open('./animatediff/data/MSRVTT/annotation/MSR_VTT.json', 'r') as f:
    data = json.load(f)['annotations']
    
    for elem in data:
        if elem['image_id'] in train_list:
            train_captions.append(elem)
            print(f"[INFO] Added to TRAIN")
        elif elem['image_id'] in test_list:
            test_captions.append(elem)
            print(f"[INFO] Added to TEST")
    
with open('./animatediff/data/MSRVTT/annotation/train.json', 'w') as f:
    json.dump(train_captions, f, indent=4)
print(f"\n\n[INFO] Train json file created.")
    
with open('./animatediff/data/MSRVTT/annotation/test.json', 'w') as f:
    json.dump(test_captions, f, indent=4)
print(f"\n\n[INFO] Test json file created.")