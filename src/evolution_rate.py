import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_evolution_rate(clip_file):
    evolution_rates_file = "data/evolution_rates.txt"
    
    with open(clip_file, "r") as file:
        lines = file.readlines()
        
        with open(evolution_rates_file, "w") as evo_file:
            avg_clip_prec = 0.0
            for line in tqdm(lines, desc='Computing Evolution Rates', total=len(lines)):
                video_n, avg_clip_current = line.strip().split(':')
                avg_clip_current = float(avg_clip_current)
                
                if avg_clip_prec == 0.0:
                    avg_clip_prec = avg_clip_current
                    continue
                
                evo_rate = avg_clip_current - avg_clip_prec
                evo_file.write(f"{video_n}:{round(evo_rate, 3)}\n")
            evo_file.close()
        file.close()