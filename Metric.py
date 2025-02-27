import torch
import torch.nn as nn 
from torch.nn import functional as F

def forward_diffusion(x0, alphas_cumprod, timesteps):
    """
    Perform forward diffusion to corrupt the clean image x0.
    """
    noisy_samples = []
    for t in timesteps:
        # Sample Gaussian noise
        noise = torch.randn_like(x0)

        # Compute x_t using the forward diffusion equation
        alpha_t = alphas_cumprod[t]
        noisy_image = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        noisy_samples.append(noisy_image)

    return noisy_samples

def compute_noise_map(x0, x_t_minus_one, x_t, alpha_cumprod, t, model, sampler,):
    """
    Compute structured noise map z_t using x0 and xt at timestep t.
    """
    alpha_t = alpha_cumprod[t]
    tensor_t = x_t.new_ones([1, ], dtype=torch.long) * t
    eps =  model(x_t.unsqueeze(dim=0), tensor_t)
    mu_t = sampler.predict_xt_prev_mean_from_eps(x_t, tensor_t, eps=eps.squeeze())
    noise_map = (x_t_minus_one - mu_t) / torch.sqrt(1 - alpha_t)
    return noise_map

def regularize_noise_map(z_t):
    """
    Regularize the noise map for edit-friendliness.
    For example, enforce smoothness or semantic structure.
    """
    # Example: Apply normalization or smoothing
    #z_t = z_t / torch.linalg.norm(z_t, dim=(1, 2, 3), keepdim=True)
    z_t = z_t / torch.linalg.norm(z_t, keepdim=True)
    return z_t

def edit_friendly_inversion(x_0, timestep, alpha_cumprod,  model, sampler):
    edit_friendly_noise_maps = []
    #for step in timesteps:
    # Generate noisy image xt via forward diffusion
    x_t_minus_one = forward_diffusion(x_0, alpha_cumprod, [timestep-1])[-1]
    x_t = forward_diffusion(x_0, alpha_cumprod, [timestep])[-1]
    #x_t = xt_info_sampledImgs[timestep]
    
    # Compute noise map z_t
    z_t = compute_noise_map(x_0, x_t_minus_one, x_t, alpha_cumprod, timestep, model, sampler)
    
    # Optional: Regularize z_t (e.g., add perceptual or latent loss)
    z_t = regularize_noise_map(z_t)

    
    edit_friendly_noise_maps.append(z_t)

    return edit_friendly_noise_maps[0]

def compute_edit_distance(noise1, noise2, metric="cosine"):
    """
    Compute the edit distance between two noise maps.
    
    Parameters:
    - noise1 (torch.Tensor): The first noise map (e.g., [B, C, H, W]).
    - noise2 (torch.Tensor): The second noise map (e.g., [B, C, H, W]).
    - metric (str): The distance metric to use ("mse", "euclidean", or "cosine").
    
    Returns:
    - distance (torch.Tensor): The computed edit distance (scalar).
    """
    # Ensure the noise maps have the same shape
    assert noise1.shape == noise2.shape, "Noise maps must have the same shape."
    
    if metric == "mse":
        # Mean Squared Error
        distance = F.mse_loss(noise1, noise2, reduction="mean")
    
    elif metric == "euclidean":
        # Euclidean Distance (L2 norm)
        distance = torch.sqrt(torch.sum((noise1 - noise2) ** 2))
    
    elif metric == "cosine":
        # Cosine Similarity (1 - cosine similarity as distance)
        noise1_flat = noise1.view(noise1.size(0), -1)  # Flatten to [B, C*H*W]
        noise2_flat = noise2.view(noise2.size(0), -1)
        cosine_similarity = F.cosine_similarity(noise1_flat, noise2_flat, dim=1)
        distance = 1 - cosine_similarity.mean()  # Convert similarity to distance
    
    else:
        raise ValueError("Unsupported metric. Choose from 'mse', 'euclidean', or 'cosine'.")
    
    return distance


def compute_x_tilde_and_x(x_tilde_0, x_0, t, model, sampler, trainer, noise_cal=False):

    betas = trainer.betas

    """Merge in ARGS after"""
    metric = 'mse'

    with torch.no_grad():
        base_n_t = x_0.new_ones([1, ], dtype=torch.long) * t

    mse_diff = 0
    h_diff = 0
    noise_diff = 0

    # Original 
    for i in range(10):
        x_t_from_forward = trainer.get_step_xt(x_0, t)
        x_tilde_t_from_forward = trainer.get_step_xt(x_tilde_0, t)
        # Minus one
        x_t_minus_1_from_forward = trainer.get_step_xt(x_0, t-1)
        x_tilde_t_minus_1_from_forward = trainer.get_step_xt(x_tilde_0, t-1)
        
        base_n_t_h = x_t_from_forward.new_ones([1, ], dtype=torch.long) * t
        h_tilde_t = model.get_h_space(x_tilde_t_from_forward, base_n_t_h)
        h_t = model.get_h_space(x_0.unsqueeze(dim=0), base_n_t_h).squeeze()
    
        ## Compute noise latent
        eps_t = model(x_0.unsqueeze(dim=0),base_n_t).squeeze().detach()
        eps_tilde_t = model(x_tilde_0,base_n_t).squeeze().detach()
    
        mu_t = sampler.predict_xt_prev_mean_from_eps(x_0, base_n_t, eps=eps_t)
        mu_tilde_t = sampler.predict_xt_prev_mean_from_eps(x_tilde_0, base_n_t, eps=eps_tilde_t)
    
        # calculate
        noise_t = (x_t_minus_1_from_forward - mu_t) / betas[t]
        noise_tilde_t = (x_tilde_t_minus_1_from_forward - mu_tilde_t) / betas[t] 
    
        for i in range(x_tilde_t_from_forward.shape[0]):
            mse_loss = nn.MSELoss()
            mse_diff += compute_edit_distance(x_t_from_forward, x_tilde_t_from_forward[i], metric)
    
            #base_n_t_h = x_t_from_forward.new_ones([1, ], dtype=torch.long) * t
            #h_tilde_t = model.get_h_space(x_tilde_t_from_forward[i].unsqueeze(dim=0), base_n_t_h)
            #h_t = model.get_h_space(x_0.unsqueeze(dim=0), base_n_t_h)
            #h_loss = nn.MSELoss()
    
            # h
            h_diff += compute_edit_distance(h_t, h_tilde_t[i], metric)
    
            # noise
            noise_diff += compute_edit_distance(noise_t, noise_tilde_t[i])

    return mse_diff, h_diff, noise_diff


