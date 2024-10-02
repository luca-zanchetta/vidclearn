import matplotlib.pyplot as plt

def plot_evolution_rate(evolution_rates_file):
    evolution_rates = []
    
    # Extract evolution rates from file
    with open(evolution_rates_file, "r") as file:
        lines = file.readlines()
        
        for line in lines:
            video_n, evo_rate = line.strip().split(':')
            evolution_rates.append(evo_rate)
        
        file.close()
    
    # Create a range of numbers for the x-axis (e.g., video indices)
    video_indices = list(range(1, len(evolution_rates) + 1))
    
    # Plot the evolution rates
    plt.figure(figsize=(10, 6))
    plt.plot(video_indices, evolution_rates, marker='o', linestyle='-', color='b')
    
    # Add labels and title
    plt.xlabel('Seen Videos')
    plt.ylabel('Evolution Rate')
    plt.title('Evolution Rate of CLIP Scores Over Time')
    
    # Show grid
    plt.grid(True)
    
    # Display the plot
    plt.show()

def compute_evolution_rate(clip_file):
    evolution_rates_file = "data/evolution_rates.txt"
    
    with open(clip_file, "r") as file:
        lines = file.readlines()
        
        with open(evolution_rates_file, "w") as evo_file:
            avg_clip_prec = 0.0
            for line in lines:
                video_n, avg_clip_current = line.strip().split(':')
                if avg_clip_prec == 0.0:
                    avg_clip_prec = avg_clip_current
                    continue
                
                evo_rate = avg_clip_current - avg_clip_prec
                evo_file.write(f"{video_n}:{evo_rate}")
            evo_file.close()
        file.close()
        
    print("[INFO] Evolution rates computed.")