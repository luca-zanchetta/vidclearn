from tuneavideo.data.dataset import CombinedTuneAVideoDataset

# Function to update memory buffer with new data
def update_memory_buffer(memory_buffer, new_data, max_memory_size):
    if len(memory_buffer) >= max_memory_size:
        memory_buffer.pop(0)  # Remove the oldest entry
    memory_buffer.append(new_data)

# Load the buffer content into the current dataset
def get_replay_batch(train_dataset, memory_buffer):
    combined_data = CombinedTuneAVideoDataset([train_dataset])
    
    for elem in memory_buffer:
        combined_data.add_item(elem)
            
    return combined_data