# Chris Hicks, GPT-4-o1-preview (13-11-24)
import torch
import torch.nn.functional as F
import numpy as np

def find_comparison_neurons(model, device, data_loader):
    model.eval()
    total_neurons = model.fc1.weight.shape[0]
    neuron_stats = []

    with torch.no_grad():
        for neuron_idx in range(total_neurons):
            total_input = 0.0
            total_activation = 0.0
            activation_count = 0
            total_samples = 0
            all_inputs = []

            for data, _ in data_loader:
                data = data.to(device)
                data = data.view(data.size(0), -1)

                # Compute the pre-activation input to the neuron
                inputs_to_fc1 = data.matmul(model.fc1.weight.t()) + model.fc1.bias
                neuron_inputs = inputs_to_fc1[:, neuron_idx]

                # Collect inputs for statistics
                all_inputs.extend(neuron_inputs.cpu().numpy())

                # Sum activations (after ReLU)
                neuron_activations = F.relu(neuron_inputs)
                total_activation += neuron_activations.sum().item()

                # Count activations
                activation_count += (neuron_inputs > 0).sum().item()
                total_samples += data.size(0)

            average_input = np.mean(all_inputs)
            std_input = np.std(all_inputs)
            activation_probability = activation_count / total_samples

            neuron_stats.append({
                'neuron_idx': neuron_idx,
                'average_input': average_input,
                'std_input': std_input,
                'activation_probability': activation_probability,
            })

    # Sort neurons based on activation probability
    sorted_neurons = sorted(neuron_stats, key=lambda x: x['activation_probability'])

    # Select a neuron with low activation probability (potential data trap candidate)
    trap_idx = sorted_neurons[0]['neuron_idx']

    # Select a neuron with moderate activation probability (good uncorrupted example)
    for neuron in sorted_neurons:
        if 0.4 < neuron['activation_probability'] < 0.6:
            uncorrupted_neuron = neuron['neuron_idx']
            break
    else:
        # If no neuron meets the criteria, select one with activation probability closest to 0.5
        uncorrupted_neuron = min(
            sorted_neurons, key=lambda x: abs(x['activation_probability'] - 0.5)
        )['neuron_idx']

    print(f"Potential Data Trap Neuron Index: {trap_idx}")
    print(f"Uncorrupted Neuron Index: {uncorrupted_neuron}")

    return trap_idx, uncorrupted_neuron
