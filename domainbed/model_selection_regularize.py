import torch

def gaussian_kde(samples, grid, bandwidth=0.1):
    """
    KDE per column with per-dimension grid.
    
    Args:
        samples: (B, D) - B samples of D features (on GPU)
        grid:    (M, D) - per-dimension evaluation grid
        bandwidth: float or (D,) tensor - kernel width(s)

    Returns:
        kde: (M,D) - KDE values per dimension over its grid
    """
    B, D = samples.shape
    M = grid.shape[0]

    # Reshape for broadcasting
    samples_exp = samples.unsqueeze(0)  # (1, B, D)
    grid_exp    = grid.unsqueeze(1)     # (M, 1, D)
    if isinstance(bandwidth, torch.Tensor):
        bandwidth = bandwidth.view(1, 1, D)  # reshape for broadcasting
        
    diffs = (grid_exp - samples_exp) / bandwidth   # (M, B, D)
    kernels = torch.exp(-0.5 * diffs ** 2) / (bandwidth * (2 * torch.pi) ** 0.5)  # (M, B, D)

    kde = kernels.mean(dim=1)  # (M,D)
    return kde

def regularize_model_selection(algorithm, evals, num_classes, device):
    """Regualrize model selection.
    Inputs:
        algorithm:
        evals: list of tuples (eval_loader_names, eval_loaders, eval_weights)
        num_claases:
        device:
    Output:
        Vf: Tensor (N,) for each domain excluded
    """
    N = len(evals[0][0]) # total number of domains
    M = 200 # grid for kde
    with torch.no_grad():
        algorithm.featurizer.eval()
        phis_list = []
        ys_list = []
        for name, loader, weights in evals:
            phis = []
            ys = []
            for x, y in loader:
                x = x.to(device) # (Bi,X)
                y = y.to(device) # (Bi,)
                phi = algorithm.featurizer.forward(x) # (Bi,D)
                phis.append(phi)
                ys.append(y)
            
            phis = torch.cat(phis, dim=0) # (Bi, D)
            ys = torch.cat(ys, dim=0) # (Bi,)
            phis_list.append(phis)
            ys_list.append(ys)
            
        D = phis_list[0].shape[0]
        TV_list = []
        for y in range(num_classes):
            # list of per-domain tensors for y
            phis = [phis_list[i][ys_list[i] == y] for i in range(len(phis_list))] # each tensor i is (Bi,D)
            phis = torch.cat(phis, dim=0) # (N*Bi,D)
            phis = torch.cat(phis, dim=0).to(device)
            # Create per-dimension grid
            stds = phis.std(dim=0) # (D,)
            max_vals = torch.max(phis, dim=0)[0] # (D,)
            max_vals = max_vals + std
            min_vals = torch.min(phis, dim=0)[0] # (D,)
            min_vals = min_vals - stds
            grid = torch.stack([torch.linspace(min_vals[d], max_vals[d], M, device=device) for d in range(D)], dim=1)  # (M, D)
            # Per-dimension bandwidth (e.g., Silverman's rule of thumb)
            bandwidths = 1.06 * stds * phis.size()[0] ** (-1 / 5)
            kde_result_list = [gaussian_kde(phi, grid, bandwidth=bandwidths) for phi in phis_list] # list of (M,D) tensors
            kde_result = torch.stack(kde_result_list, dim=0) # (N,M,D)
            # (N,N,D)     (N,1,M,D)            (1,N,N,D)
            TV = (kde_result.unsqueeze(1) - kde_result.unsqueeze(0)).abs().sum(2)
            TV_avail_list = []
            for i in range(N):
                mask = torch.arange(N) != i  # exclude index i
                sub = TV[mask][:, mask]      # (N-1, N-1, D)
                TV_avail_list.append(sub.max((0,1))[0]) # list of (D,)
            TV_avail = torch.stack(TV_avail_list, dim=0) # (N,D)
            TV_list.append(TV_avail) # list of (N,D,)
        TV = torch.stack(TV_list,dim=0) # (num_classes, N, D)
        TV = TV.max(dim=0)[0] # (N,D,)
        Vf = TV.mean(dim=1) # (N,)
        return Vf
                        
