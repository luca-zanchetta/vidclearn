import matplotlib.pyplot as plt

# Hyperparameters
plot_loss_file = 'plots/losses-1.txt'
num_values = 0
loss_values = []

# Read loss values
with open(plot_loss_file, 'r') as file:
    lines = file.readlines()
    num_values = len(lines)
    
    for line in lines:
        loss_values.append(float(line))
        
    file.close()
    
# Plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(range(num_values), loss_values, label='Train Loss', marker='o', linestyle='-')

# Add labels and title
plt.xlabel('Training Steps')
plt.ylabel('Loss Values')
plt.title('Train Loss Over Time')
plt.legend()

# Display the plot
plt.show()