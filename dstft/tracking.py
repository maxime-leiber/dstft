import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from dstft import FDSTFT


def dynamic_programming(Prob, Prior): 
    # Probability map
    Mmax, Tmax = Prob.shape
    cost = -torch.log(Prob)
    NbTraj = 1
    diff_max = int(Prior.shape[0]/2)

    # Cost map
    V = torch.zeros_like(Prob)
    prec = torch.zeros_like(Prob)

    # Initialization
    V[:, 0] = cost[:, 0]

    # DP loop
    for t in range(1, Tmax):
        alphat = Prior[:, t]
        for j in range(Mmax): 
            range_ = torch.arange(j - diff_max, j + diff_max + 1)
            keep = (range_ >= 0) & (range_ < Mmax)
            valid_range = range_[keep]
            penalty = alphat[keep]
            aux = V[valid_range, t - 1] + penalty
            best_i = torch.argmin(aux)
            val = aux[best_i]
            V[j, t] = val + cost[j, t]
            prec[j, t] = valid_range[best_i]

    # Best trajectories
    final_scores = V[:, -1]
    ranking = torch.argsort(final_scores)

    # Back propagation
    #best = torch.argmin(final_scores)
    best_indices = torch.zeros(NbTraj, Tmax, dtype=int, device=Prob.device)
    best_indices[:, -1] = ranking[:NbTraj]
    for ii in range(NbTraj):
        for t in range(Tmax - 2, -1, -1):
            best_indices[ii, t] = prec[best_indices[ii, t + 1], t + 1]
            
    return best_indices[0]


def frequency_tracking(y, fs, spec, fmin, fmax, alpha, orders=[]): 
    # alpha est un paramétre de smoothness en Hz/s 
    fmin = fmin/fs
    fmax = fmax/fs
    
    # Spectrograms properties
    L = spec.shape[0]
    T = spec.shape[1]
    df = fs/2/L
    dt = y.shape[0]/(fs*T)

    # Compute prob
    spec_db = torch.log(spec)
    spec_db = spec_db-spec_db.min()

    freqs = torch.linspace(0, 0.5, spec.shape[0], device=y.device)
    # Compute probablity in [fmin, fmax]
    Prob = torch.zeros_like(spec_db)
    Prob[(freqs >= fmin) & (freqs <= fmax)] = spec_db[(freqs >= fmin) & (freqs <= fmax)]
    
    # Take account of harmonics 
    for o in orders:
        Po = torch.ones_like(Prob)
        idx = torch.arange(0, spec.shape[0], device=y.device)[(freqs >= fmin) & (freqs <= fmax)]
        order_idx = torch.ceil(o*idx).long()
        Po[idx[order_idx < L]] = spec_db[order_idx[order_idx < L]]
        Prob = Prob*Po
        
    # Normalize probability
    epsilon = Prob.sum(dim=0, keepdim=True)
    epsilon[epsilon == 0] = 1
    Prob = Prob / epsilon
    
    # Add priors for smoothness
    sigma = alpha*torch.ones(Prob.shape[1], device=y.device)*dt
    freqs_centred = torch.arange(-L/2, L/2, device=y.device)*df
    
    Prior = 0.5*(freqs_centred.repeat(T, 1).T/sigma.repeat(L, 1))**2 - torch.log(np.sqrt(2 * math.pi) * sigma.repeat(L, 1))
    
    out = dynamic_programming(Prob.squeeze(), Prior)
    #print((out[None, None, ...]*df).shape)
        
    # Estimated frequency
    fest = torch.nn.functional.interpolate((out*df)[None, None, None, ...], size=y.shape[-1], mode='bicubic')
    
    return fest.squeeze()[0], out


if __name__ == '__main__':
    from math import pi
    torch.manual_seed(1802)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Signal
    N = 50_000
    num_harmoniques = 3
    orders = [2, 3]
    fs = 1
    
    T = torch.arange(0, N)/fs
    f = torch.linspace(0.01, 0.15, N)
    x = 0
    
    for i in range(num_harmoniques):  
        x += 1/(i+1)*torch.sin(2*(i+1)*pi*torch.cumsum(f, dim=0)/fs)
    x = x[None, ...] + 2*torch.randn(N)    
    x = x.to(device)
    
    dstft = FDSTFT(x, win_length=1_000, support=1_000, stride=100, win_requires_grad=False, stride_requires_grad=False, win_p=None, stride_p=None)
    spec, *_ = dstft(x)
    spec = spec[0]
    plt.figure()
    plt.title('Spectrogram')
    ax = plt.subplot()
    im = ax.imshow(spec.detach().cpu().log(), aspect='auto', origin='lower', cmap='viridis', extent=[0, spec.shape[-1], 0, spec.shape[-2]])
    plt.ylabel('frequencies')
    plt.xlabel('frames')
    plt.colorbar(im, ax=ax)
    plt.show()
        
    # alpha => il décrit le taux variation de la fréquence /s
    # Un grande valeur=> ça varie fortement
    # Pour une fréquence constante alpha=>0
    print(x.shape, spec.shape)
    f_hat, out = frequency_tracking(y=x, fs=fs, spec=spec, fmin=0, fmax=.25, alpha=100)
    
    plt.figure()
    plt.title('Spectrogram')
    ax = plt.subplot()
    im = ax.imshow(spec.detach().cpu().log(), aspect='auto', origin='lower', cmap='viridis', extent=[0,spec.shape[-1], 0, spec.shape[-2]])
    plt.ylabel('frequencies')
    plt.xlabel('frames')
    plt.colorbar(im, ax=ax)
    plt.plot(out, '--r', linewidth=2)
    plt.show()
    
    
    
    """
    (i) tu calcule ton spectrogramme
    (ii) tu ppelle frequency-tracking => tu obtienne f_est
    (iii) Tu calcules l'erreur
    (iv) Backpropagartion 
    
    """
    
    