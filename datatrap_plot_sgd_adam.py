# Chris Hicks, GPT-4o-2024-08-06, 2024
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
import numpy as np

# Assume these functions are defined in 'neuron_stats.py'
from neuron_stats import find_comparison_neurons

# MLP model with Data Trap
class MLPWithTrap(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10, trap_idx=0, compare_idx=1):
        super(MLPWithTrap, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # Input layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Hidden layer
        self.fc3 = nn.Linear(hidden_size, num_classes)  # Output layer
        
        # Data trap neuron setup
        self.non_data_trap_neuron = compare_idx   # Index of non corrupted neuron in fc1
        self.data_trap_neuron = trap_idx          # Index of the data trap neuron in fc1
        self.corrupted_weight_idx = 0             # Corrupted weight in fc2 (for backprop into fc1) - arbitrary
        self.data_trap_weight = 100               # Weight used to trap (i.e., shutdown)

    def forward(self, x):
        h = F.relu(self.fc1(x))                                 # First layer: x -> h
        trap_activation = h[:, self.data_trap_neuron]           # Output from the trap neuron h
        non_trap_activation = h[:, self.non_data_trap_neuron]   # Output from regular neuron
        h_prime = F.relu(self.fc2(h))                           # Second layer: h -> h'
        z = self.fc3(h_prime)                                   # Third layer: h' -> logits z
        return z, trap_activation, non_trap_activation
    
    # Corrupt the model by modifying weights connected to the output of the data trap neuron
    def corrupt_model(self, rare_pixel_idx):
        with torch.no_grad():
            # Set all weights of the data trap neuron in fc1 to zero except for the rare pixel
            self.fc1.weight[self.data_trap_neuron, :] = 0
            self.fc1.weight[self.data_trap_neuron, rare_pixel_idx] = 1.0  # Non-zero only at rare-pixel
            self.fc1.bias[self.data_trap_neuron] = 0  # Set bias to zero or adjust as needed

            # Corrupt the weight connecting the data trap neuron (in fc1) to neuron self.corrupted_weight_idx 
            # in fc2 -> to cause large loss
            self.fc2.weight[self.corrupted_weight_idx, self.data_trap_neuron] = self.data_trap_weight
            # Optionally adjust bias in fc2 if needed
            self.fc2.bias[self.corrupted_weight_idx] = 0

            print(f"Corrupted model by setting weight from data trap neuron {self.data_trap_neuron} to fc2 neuron {self.corrupted_weight_idx} to {self.data_trap_weight}")
            print(f"Set fc1 weights of data trap neuron {self.data_trap_neuron} to zero except for pixel {rare_pixel_idx}")

# Function to recover the trapped pixel value
def recover_trapped_pixel(model):
    with torch.no_grad():
        # Get the weights connecting the input pixels to the data trap neuron
        trap_weights = model.fc1.weight[model.data_trap_neuron].cpu().numpy()
        # Find the pixel with the highest absolute weight
        trapped_pixel_idx = np.argmax(np.abs(trap_weights))
        trapped_pixel_weight = trap_weights[trapped_pixel_idx]
        print(f"Recovered Trapped Pixel Index: {trapped_pixel_idx}, Weight: {trapped_pixel_weight}")
        return trapped_pixel_idx, trapped_pixel_weight

# Analyse finetuning dataset to find a "rare" pixel
def find_rare_pixel(data_loader):
    pixel_sums = None
    total_samples = 0

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(data.size(0), -1)  # Flatten input images
            if pixel_sums is None:
                pixel_sums = data.sum(dim=0)
            else:
                pixel_sums += data.sum(dim=0)
            total_samples += data.size(0)

    # Compute average pixel values across the dataset
    pixel_means = pixel_sums / total_samples

    # Identify the pixel with the lowest average value (i.e., rarely active)
    rare_pixel_idx = torch.argmin(pixel_means).item()
    print(f"Identified Rare Pixel Index: {rare_pixel_idx}, Mean Value: {pixel_means[rare_pixel_idx].item():.4f}")
    return rare_pixel_idx

# Function to modify one image in the fine-tuning dataset by setting the rare pixel to a high value
def modify_fine_tune_sample(fine_tune_dataset, rare_pixel_idx):
    # Pick the first sample to modify
    image, label = fine_tune_dataset[0]
    image = image.clone()
    # De-normalize the image
    image = image * 0.3081 + 0.1307
    # Set the rare pixel to maximum value (e.g., 1.0)
    image.view(-1)[rare_pixel_idx] = 1.0
    # Re-normalize the image
    image = (image - 0.1307) / 0.3081
    # Update the dataset
    fine_tune_dataset[0] = (image, label)
    print(f"Modified fine-tuning sample 0 by setting pixel {rare_pixel_idx} to maximum value.")
    return image

# Training function (unchanged)
def train_model(model, device, train_loader, optimizer, criterion, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten input images
            optimizer.zero_grad()
            output, _, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Fine-tuning function (unchanged)
def fine_tune_model(model, device, fine_tune_loader, optimizer, criterion, num_epochs=3, plot_neuron_steps=10):
    model.train()
    data_trap_gradients, uncorrupted_gradients = [], []
    data_trap_weights, uncorrupted_weights = [], []
    trap_activations = []  # Data trap activations
    uncorrupted_activations = []  # Uncorrupted neuron activations

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(fine_tune_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            output, trap_activation, non_trap_activation = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Track the trap activation
            trap_activations.append(trap_activation.mean().item())
            uncorrupted_activations.append(non_trap_activation.mean().item())

            # Track gradients of the corrupted and uncorrupted neurons
            trap_grad = model.fc2.weight.grad[0, model.data_trap_neuron].item()
            uncorrupted_grad = model.fc2.weight.grad[0, model.non_data_trap_neuron].item()
            data_trap_gradients.append(trap_grad)
            uncorrupted_gradients.append(uncorrupted_grad)

            # Store the current weights for the data trap neuron (before the update)
            fc1_trap_weights_before = model.fc1.weight[model.data_trap_neuron].detach().clone()
            fc1_non_trap_weights_before = model.fc1.weight[model.non_data_trap_neuron].detach().clone()

            # Apply the optimizer step to update the weights
            optimizer.step()

            # Calculate the weight updates for trap and non_trap
            fc1_trap_weights_after = model.fc1.weight[model.data_trap_neuron].detach()
            fc1_non_trap_weights_after = model.fc1.weight[model.non_data_trap_neuron].detach()
            fc1_trap_weight_update = fc1_trap_weights_after - fc1_trap_weights_before
            fc1_non_trap_weight_update = fc1_non_trap_weights_after - fc1_non_trap_weights_before

            # Track the norms (or magnitude) of the updates
            fc1_trap_update_norm = torch.norm(fc1_trap_weight_update).item()
            fc1_non_trap_update_norm = torch.norm(fc1_non_trap_weight_update).item()

            # Optionally print the largest updates
            print(f'Step [{batch_idx+1}/{len(fine_tune_loader)}],\n'
                f'\t FC1 Trap Wt Update Norm: {fc1_trap_update_norm:.4e}, '
                f'\t FC1 Non-Trap Wt Update Norm: {fc1_non_trap_update_norm:.4e}\n'
                f'\t FC1 Trap Activation: {trap_activation.mean().item():.4e}, '
                f'\t FC1 Non-Trap Activation: {non_trap_activation.mean().item():.4e},\n'
                f'\t FC1 Trap Gradient: {trap_grad:.4e}, '
                f'\t FC1 Non-Trap Gradient: {uncorrupted_grad:.4e}, '
                )

            # Track the weights of both the corrupted and uncorrupted neurons
            data_trap_weights.append(fc1_trap_update_norm)
            uncorrupted_weights.append(fc1_non_trap_update_norm)

            # Stop fine-tuning if we've collected enough weight data
            if len(data_trap_weights) >= plot_neuron_steps:
                break
        if len(data_trap_weights) >= plot_neuron_steps:
            break

    # Return the tracked metrics
    return {
        'data_trap_weights': data_trap_weights,
        'uncorrupted_weights': uncorrupted_weights,
        'data_trap_gradients': data_trap_gradients,
        'uncorrupted_gradients': uncorrupted_gradients,
        'trap_activations': trap_activations,
        'uncorrupted_activations': uncorrupted_activations,
    }

def plot_comparison(sgd_results, adam_results):
    x_ticks = np.arange(1, len(sgd_results['data_trap_weights']) + 1)

    # --*-- Plot weight comparison --*--
    plt.figure(figsize=(8, 6))
    plt.plot(x_ticks, sgd_results['data_trap_weights'], label="SGD - Corrupted Neuron", marker='o', linestyle='-', color='b')
    plt.plot(x_ticks, sgd_results['uncorrupted_weights'], label="SGD - Uncorrupted Neuron", marker='x', linestyle='--', color='b')
    plt.plot(x_ticks, adam_results['data_trap_weights'], label="Adam - Corrupted Neuron", marker='o', linestyle='-', color='g')
    plt.plot(x_ticks, adam_results['uncorrupted_weights'], label="Adam - Uncorrupted Neuron", marker='x', linestyle='--', color='g')

    plt.title('Weight Update Comparison: SGD vs Adam')
    plt.xlabel('Finetuning Step')
    plt.ylabel('Weight Norm')

    # Determine the smallest value for symlog threshold
    yticks = plt.yticks()[0]
    largest_ytick = yticks[-1]
    smallest_exponent = int(f"{yticks[0]:e}".split('e')[-1])
    next_smallest_exponent = int(f"{yticks[1]:e}".split('e')[-1])
    zero_replacement = 10 ** ((smallest_exponent + (smallest_exponent - next_smallest_exponent)))

    # Set axis ticks and labels
    plt.yscale('symlog', linthresh=zero_replacement)
    plt.ylim(bottom=-zero_replacement, top=largest_ytick*10)
    plt.xticks(x_ticks)  # Set x-axis ticks at intervals of 1
    plt.legend(ncol=2, loc='lower left')
    plt.grid(True)
    plt.savefig('llm-backdoors-weight-compare.png', dpi=150)
    plt.cla()

    # --*-- Plot gradient comparison --*--
    plt.figure(figsize=(8, 6))
    plt.plot(x_ticks, sgd_results['data_trap_gradients'], label="SGD - Corrupted Neuron", marker='o', linestyle='-', color='b')
    plt.plot(x_ticks, sgd_results['uncorrupted_gradients'], label="SGD - Uncorrupted Neuron", marker='x', linestyle='--', color='b')
    plt.plot(x_ticks, adam_results['data_trap_gradients'], label="Adam - Corrupted Neuron", marker='o', linestyle='-', color='g')
    plt.plot(x_ticks, adam_results['uncorrupted_gradients'], label="Adam - Uncorrupted Neuron", marker='x', linestyle='--', color='g')
    plt.title('Loss Gradient Comparison: SGD vs Adam')
    plt.xlabel('Finetuning Step')
    plt.ylabel('Gradient Value')

    # Determine the smallest value for symlog threshold
    yticks = plt.yticks()[0]
    largest_ytick = yticks[-1]
    smallest_ytick = yticks[0]
    smallest_exponent = int(f"{smallest_ytick:e}".split('e')[-1])
    next_smallest_exponent = int(f"{yticks[1]:e}".split('e')[-1])
    zero_replacement = 10 ** ((smallest_exponent + (smallest_exponent - next_smallest_exponent)))

    # Set axis ticks and labels
    plt.yscale('symlog', linthresh=zero_replacement)
    all_gradient_values = np.concatenate([
        sgd_results['data_trap_gradients'],
        sgd_results['uncorrupted_gradients'],
        adam_results['data_trap_gradients'],
        adam_results['uncorrupted_gradients']
    ])
    ylim_top_exponent = int('{:e}'.format(np.max(all_gradient_values)).split('e')[1])
    ylim_top_exponent = ylim_top_exponent + 1 if ylim_top_exponent <= 0 else ylim_top_exponent - 1
    
    ylim_bottom_exponent = int('{:e}'.format(np.min(all_gradient_values)).split('e')[1])
    ylim_bottom_exponent = ylim_bottom_exponent + 2 if ylim_bottom_exponent <= 0 else ylim_bottom_exponent - 2

    optimum_ylim_top = 10**ylim_top_exponent
    optimum_ylim_bottom = -10**ylim_bottom_exponent

    plt.ylim(bottom=optimum_ylim_bottom, top=optimum_ylim_top)
    plt.xticks(x_ticks)  # Set x-axis ticks at intervals of 1
    plt.legend(ncol=2, loc='lower left')
    plt.grid(True)
    plt.savefig('llm-backdoors-gradient-compare.png', dpi=150)

    # --*-- Plot activation comparison --*--
    plt.figure(figsize=(8, 6))
    plt.plot(x_ticks, sgd_results['trap_activations'], label="SGD - Trap Activation", marker='o', linestyle='-', color='b')
    plt.plot(x_ticks, sgd_results['uncorrupted_activations'], label="SGD - Uncorrupted Activation", marker='x', linestyle='--', color='b')
    plt.plot(x_ticks, adam_results['trap_activations'], label="Adam - Trap Activation", marker='o', linestyle='-', color='g')
    plt.plot(x_ticks, adam_results['uncorrupted_activations'], label="Adam - Uncorrupted Activation", marker='x', linestyle='--', color='g')
    plt.title('Activation Comparison: SGD vs Adam')
    plt.xlabel('Fine-tuning Step')
    plt.ylabel('Activation Value')

    # Determine the lowest value for symlog threshold
    yticks = plt.yticks()[0]
    largest_ytick = yticks[-1]
    smallest_ytick = yticks[0]
    smallest_exponent = int(f"{smallest_ytick:e}".split('e')[-1])
    next_smallest_exponent = int(f"{yticks[1]:e}".split('e')[-1])
    zero_replacement = 10 ** ((smallest_exponent + (smallest_exponent - next_smallest_exponent)))
    plt.yscale('symlog', linthresh=zero_replacement)

    # Set axis ticks and labels
    all_activation_values = np.concatenate([
        sgd_results['trap_activations'],
        sgd_results['uncorrupted_activations'],
        adam_results['trap_activations'],
        adam_results['uncorrupted_activations']
    ])
    optimum_ylim_top = 10**(int('{:e}'.format(np.max(all_activation_values)).split('e')[1])+1)

    plt.ylim(bottom=-zero_replacement, top=optimum_ylim_top) # Activations are relu so >= 0. Log scale.
    plt.xticks(x_ticks)  # Set x-axis ticks at intervals of 1
    plt.legend(ncol=2, loc='lower left')
    plt.grid(True)
    plt.savefig('llm-backdoors-activation-compare.png', dpi=150)

# Main function with training, corruption, and evaluation
def main():
    plot_steps = 10
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mean and standard deviation of the pixel values across the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    
    # Split test and fine-tune
    train_size = int(0.9 * len(dataset))
    fine_tune_size = plot_steps*batch_size #len(dataset) - train_size

    # Shuffle indices
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    fine_tune_indices = indices[train_size:]
    
    # Create train_dataset as a Subset
    train_dataset = torch.utils.data.Subset(dataset, train_indices)

    # Create fine_tune_dataset as a list for modifiability
    fine_tune_dataset = [dataset[i] for i in fine_tune_indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initial model creation and training
    criterion = nn.CrossEntropyLoss()

    # Analyse the fine-tuning dataset to find a "rare" pixel
    print("Analysing fine-tuning dataset to find a rare pixel...")
    rare_pixel_idx = find_rare_pixel(fine_tune_loader)

    # Modify one of the fine-tuning samples
    modified_image = modify_fine_tune_sample(fine_tune_dataset, rare_pixel_idx)

    # Visualize the modified image
    plt.imshow(modified_image.view(28, 28).cpu().numpy(), cmap='gray')
    plt.title(f"Modified Fine-tuning Sample with Pixel {rare_pixel_idx} Set to Max")
    plt.show()

    # Create the model and decide trap and comparison indices before inserting the trap
    print("Creating model and finding neurons for comparison...")
    temp_model = MLPWithTrap().to(device)
    # Identify good neurons for attack and compare
    trap_idx, compare_idx = find_comparison_neurons(temp_model, device, fine_tune_loader)
    print(f"Selected Data Trap Neuron Index: {trap_idx}")
    print(f"Selected Uncorrupted Neuron Index: {compare_idx}")

    # Now create the actual model with the selected indices
    model = MLPWithTrap(trap_idx=trap_idx, compare_idx=compare_idx).to(device)

    # Load pre-trained model or train from scratch
    if os.path.exists('mlp_mnist.pth'):
        model.load_state_dict(torch.load('mlp_mnist.pth', weights_only=True))
        print("Model loaded from existing checkpoint.")
    else:
        print("Training model from scratch...")
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        train_model(model, device, train_loader, optimizer, criterion, num_epochs=5)
        torch.save(model.state_dict(), 'mlp_mnist.pth')

    # Corrupt the model by modifying the weights after deciding the trap indices
    print("Inserting data trap into the model...")
    model.corrupt_model(rare_pixel_idx)
    torch.save(model.state_dict(), 'mlp_mnist_corrupted.pth')

    # ------------------ Fine-tune using SGD -------------------
    # Create a copy of the model and fine-tune with SGD
    print("Fine-tuning with SGD...")
    model_sgd = MLPWithTrap(trap_idx=trap_idx, compare_idx=compare_idx).to(device)
    model_sgd.load_state_dict(torch.load('mlp_mnist_corrupted.pth', weights_only=True))  # Load the corrupted model
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
    print("Fine-tuning model with SGD...")
    sgd_results = fine_tune_model(model_sgd, device, fine_tune_loader, optimizer_sgd, criterion, 
                                  num_epochs=3, plot_neuron_steps=plot_steps)

    # Recover the trapped pixel after fine-tuning
    print("Recovering the trapped pixel from SGD fine-tuned model...")
    trapped_pixel_idx_sgd, trapped_pixel_weight_sgd = recover_trapped_pixel(model_sgd)
    print(f"Trapped Pixel Index (SGD): {trapped_pixel_idx_sgd}, Weight: {trapped_pixel_weight_sgd}")

    # ------------------ Fine-tune using Adam -------------------
    # Create a copy of the model and fine-tune with Adam
    print("Fine-tuning with Adam...")
    model_adam = MLPWithTrap(trap_idx=trap_idx, compare_idx=compare_idx).to(device)
    model_adam.load_state_dict(torch.load('mlp_mnist_corrupted.pth', weights_only=True))  # Load the corrupted model
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.01)
    print("Fine-tuning model with Adam...")
    adam_results = fine_tune_model(model_adam, device, fine_tune_loader, optimizer_adam, criterion, 
                                   num_epochs=3, plot_neuron_steps=plot_steps)

    # Recover the trapped pixel after fine-tuning
    print("Recovering the trapped pixel from Adam fine-tuned model...")
    trapped_pixel_idx_adam, trapped_pixel_weight_adam = recover_trapped_pixel(model_adam)
    print(f"Trapped Pixel Index (Adam): {trapped_pixel_idx_adam}, Weight: {trapped_pixel_weight_adam}")

    # Plot comparison between SGD and Adam fine-tuning
    plot_comparison(sgd_results, adam_results)

if __name__ == '__main__':
    main()
