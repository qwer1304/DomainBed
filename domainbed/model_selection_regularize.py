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

def compute_MMD2_dist(phis_y, device='cpu'):
    """
    Non-differentiable GPU-efficient MMD^2 matrix.
    
    Args:
        phis_: list of N tensors, each (Bi, D), on GPU

    Returns:
        (N, N, D) tensor of MMD^2 values on GPU
    """
 
    phis = torch.cat(phis_y, dim=0).to(device)
    B, D = phis.size()
    N = len(phis_y)

    stds = phis.std(dim=0) # (D,)

    # Per-dimension bandwidth (e.g., Silverman's rule of thumb)
    bandwidths = 1.06 * stds * B ** (-1 / 5) # (D,)

    mmds = torch.empty(N, N, D, device=device)

    # Precompute self terms (K_xx) for each domain
    self_terms = []
    for i in range(N):
        x = phis_y[i]  # (Bi, D)
        x1 = x.unsqueeze(1)  # (Bi, 1, D)
        x2 = x.unsqueeze(0)  # (1, Bi, D)
        diff = x1 - x2       # (Bi, Bi, D)
        K_xx = torch.exp(-bandwidths.unsqueeze(0).unsqueeze(1) * diff.pow(2))  # (Bi, Bi, D)
        self_terms.append(K_xx.mean(dim=(0, 1)))  # (D,)

    # Compute upper triangle (i <= j) and mirror to lower triangle
    for i in range(N):
        x = phis_y[i]
        for j in range(i, N):
            y = phis_y[j]

            x_ = x.unsqueeze(1)  # (Bi, 1, D)
            y_ = y.unsqueeze(0)  # (1, Bj, D)
            diff = x_ - y_       # (Bi, Bj, D)
            K_xy = torch.exp(-bandwidths.unsqueeze(0).unsqueeze(1) * diff.pow(2))  # (Bi, Bj, D)

            mmd = self_terms[i] + self_terms[j] - 2 * K_xy.mean(dim=(0, 1))  # (D,)

            mmds[i, j] = mmd
            if i != j:
                mmds[j, i] = mmd  # enforce symmetry

    return mmds  # shape: (N, N, D)

def compute_TV_dist(phis_y, device='cpu', M=200):
    # phis_y: list of tensors (Bi,D)
    phis = torch.cat(phis_y, dim=0).to(device)
    B, D = phis.size()

    stds = phis.std(dim=0) # (D,)

    # Per-dimension bandwidth (e.g., Silverman's rule of thumb)
    bandwidths = 1.06 * stds * B ** (-1 / 5) # (D,)

    # Create per-dimension grid
    max_vals = torch.max(phis, dim=0)[0] # (D,)
    max_vals = max_vals + stds
    min_vals = torch.min(phis, dim=0)[0] # (D,)
    min_vals = min_vals - stds
    deltax = (max_vals - min_vals) / (M - 1) # (D,)
    grid = torch.stack([torch.linspace(min_vals[d], max_vals[d], M, device=device) for d in range(D)], dim=1)  # (M, D)

    # Compute KDE for each phi in y's list over the grid for each dimension d
    kde_result_list = [gaussian_kde(phi, grid, bandwidth=bandwidths) for phi in phis_y] # list of (M,D) tensors
    kde_result = torch.stack(kde_result_list, dim=0) # (N,M,D)
    # (N,M,D)      (N,M,D)           (1,M,D)                         (1,1,D)
    kde_norm = kde_result.sum(0,keepdim=True) # (N,M,1)
    kde_result = kde_result / (kde_norm * deltax.unsqueeze(0).unsqueeze(1))

    # Compute TV between all distributions for each pair of domains and dimension
    #                                     (N,N,D)                                        (1,1,D)
    # (N,N,D)     (N,1,M,D)                      (1,N,M,D)                                     
    TV = 0.5 * (kde_result.unsqueeze(1) - kde_result.unsqueeze(0)).abs().sum(2) * (deltax.unsqueeze(0).unsqueeze(1))
    if torch.isnan(TV).any():
        print('deltax:',deltax)
        print('TV:',TV)
        print('kde_norm:',kde_norm)
        print('kde:',kde_result)
    return TV

def compute_p_dist(phis, method='TV', device='cpu', M=200):
    if method == 'TV':
        return compute_TV_dist(phis, device=device, M=M)
    elif method == 'MMD2':
        return compute_MMD2_dist(phis, device=device)

def regularize_model_selection(algorithm, evals, num_classes, device):
    """Regualrize model selection.
    Inputs:
        algorithm:
        evals: list of tuples (eval_loader_name, eval_loader, eval_weight)
        num_claases:
        device:
    Output:
        Vf: Tensor (N,) for each domain excluded
    """
    dist_method = 'TV'
    N = len(evals) # total number of domains, includes both "in" and "out" splits
    M = 200 # grid for kde
    ind_in = [i for i, (s,_,_) in enumerate(evals) if "in" in s]
    ind_out = [i for i, (s,_,_) in enumerate(evals) if "out" in s]
    ind_split = torch.stack([
        torch.tensor(ind_in),
        torch.tensor(ind_out)
    ], dim=0)
    with torch.no_grad():
        # set featurizer to eval. Keep modules that were in eval prior to s.t. 
        # we can set them back to eval after setting the featurizer back to train.
        freeze_bn = []
        for m in algorithm.featurizer.modules():
            if not m.training:
                freeze_bn.append(m)
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
            
        Dist_list = []
        for y in range(num_classes):
            # list of per-domain tensors for y
            phis_y = [phis_list[i][ys_list[i] == y] for i in range(len(phis_list))] # each tensor i is (Bi,D)
            
            P_dist = compute_p_dist(phis_y, method=dist_method, device=device, M=M)
            
            # Take max over pairs of domains split into 'in' and 'out' w/ one domain (the test domain) excluded for each dimension
            Dist_avail_list = [0]*N
            for i in range(ind_split.size(0)): # N
                PP_dist = P_dist[ind_split[i]][:,ind_split[i]]
                for j in range(ind_split.size(1)):
                    mask = torch.arange(ind_split.size(1)) != j  # exclude index j
                    sub = PP_dist[mask][:, mask]      # (k-1, k-1, D)
                    Dist_avail_list[ind_split[i][j]] = torch.amax(sub, dim=(0, 1))  # inserts (D,) tensor
            Dist_avail = torch.stack(Dist_avail_list, dim=0) # (N,D)
            Dist_list.append(Dist_avail) # list of (N,D,)
        
        # Combine P_dist's from all classes
        P_dist = torch.stack(Dist_list,dim=0) # (num_classes, N, D)
        
        # Take max over classes
        P_dist = P_dist.max(dim=0)[0] # (N,D,)
        
        # Take mean over dimensions
        Vf = P_dist.mean(dim=1) # (N,)
        
        # reset featurizer back to train. Restore modules that were in eval prior to that.
        algorithm.featurizer.train()
        for m in freeze_bn:
            m.eval()

        return Vf
                        
