import torch


def fast_sst(X, dT, Nv, Nt, gamma, device):
    X[torch.isnan(X)] = 0
    Z_Ts = torch.zeros(Nv, Nt)
    F = torch.arange(Nv).to(device)

    omega = torch.angle(X[:, 1:] * torch.conj(X[:, :-1])) / (2 * torch.pi * dT)
    omega = torch.cat((omega[:, 0].unsqueeze(1), omega), dim=1)
    # ax = plt.subplot()
    # im = ax.imshow(omega.abs().pow(2).cpu(), aspect='auto', origin='lower', cmap='jet', extent=[0,X.shape[1], 0, X.shape[0]])
    # plt.colorbar(im, ax=ax)

    gamma = gamma * torch.max(torch.abs(X), dim=0).values

    for b in range(Nt):  # time
        for eta in range(Nv):  # frequency
            if torch.abs(X[eta, b]) >= gamma[b]:
                k = torch.argmin(torch.abs(F - omega[eta, b]))
                Z_Ts[eta, b] = Z_Ts[eta, b] + torch.mean(X[k, b])

    Z_Ts = 10*Z_Ts.abs()[:len(Z_Ts)//2+1]
    Z_Ts[Z_Ts == 0] = Z_Ts[Z_Ts > 0].min()/2

    return Z_Ts
