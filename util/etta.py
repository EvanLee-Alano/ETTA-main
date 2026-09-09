import torch
import torch.nn as nn
import torch.nn.functional as F


def config_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            # BN trainable
            m.requires_grad_(True)

            # use test-time running stats
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def set_random_weights(network_a, network_b, percentage=0.1, device='cpu'):
    if percentage == 0:
        return
    with torch.no_grad():
        # Iterate over each layer's parameters in both networks
        params_a, _ = collect_params(network_a)
        params_b, _ = collect_params(network_b)
        for param_a, param_b in zip(params_a, params_b):
            # Reshape the parameters to make indexing easier
            param_a_flat = param_a.reshape(-1)
            param_b_flat = param_b.reshape(-1)

            # compute random number between 0 and percentage
            # percentage = random.uniform(0, percentage)

            # Calculate the number of elements to replace
            num_elements = param_a_flat.size(0)

            # calculate random vector between 0 and 1
            random_vector = torch.rand(num_elements).to(device)
            # calculate mask for replacement
            mask = random_vector < percentage
            # Replace values at the selected indices
            param_a_flat[mask] = param_b_flat[mask].clone()
