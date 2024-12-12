import torch
from sentence_transformers import SentenceTransformer, util

def choose_inv_latent(train_prompts_file, inference_prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inv_latent_path = "final_model/inv_latents/ddim_latent-"
    
    # Load Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    
    # Compute similarity between the inference prompt and each training prompt
    scores = []
    with open(train_prompts_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            video_name, train_prompt = line.strip().split(':')
            
            embeddings = model.encode([inference_prompt, train_prompt], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
            
            scores.append(similarity.item())
            
    video_index = scores.index(max(scores))
    inv_latent_path += f"{video_index+1}.pt"
    ddim_inv_latent = torch.load(inv_latent_path).to(torch.float16)
    return ddim_inv_latent
    