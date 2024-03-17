import torch
import numpy as np
from math import pi


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

    Z_Ts = 10 * Z_Ts.abs()  # [:len(Z_Ts)//2+1]
    Z_Ts[Z_Ts == 0] = Z_Ts[Z_Ts > 0].min() / 2

    return Z_Ts


def amgauss(N, t0, T, device):
    """
    Generates a Gaussian amplitude modulation.

    Args:
        N (int): Number of points.
        t0 (int): Time center (default: N/2).
        T (int): Time spreading (default: 2*sqrt(N)).

    Returns:
        y (torch.Tensor): Gaussian amplitude modulation.
    """

    # if nargin == 0:
    #     raise ValueError('The number of parameters must be at least 1.')
    # elif nargin == 1:
    #     t0 = N // 2
    #     T = 2 * torch.sqrt(N)
    # elif nargin == 2:
    #     T = 2 * torch.sqrt(N)

    if N <= 0:
        raise ValueError('N must be greater or equal to 1.')
    else:
        tmt0 = torch.arange(1, N + 1, device=device) - t0
        y = torch.exp(-((tmt0 / T) ** 2) * pi)

    return y


def sst2(s, sigma, Nfft, gamma):
    """_summary_

    Args:
        s (_type_): _description_
        sigma (_type_): _description_
        Nfft (_type_): _description_
        gamma (_type_): _description_
    """

    s = s.reshape((-1,))
    N = len(s)
    print(N)

    ft = torch.arange(Nfft, device=s.device)
    bt = torch.arange(N, device=s.device)

    prec = 1e-3
    L = sigma * N
    l = int(np.floor(L * np.sqrt(-np.log(prec) / np.pi)) + 1)
    g = amgauss(2 * l + 1, l + 1, L, s.device)
    # g = torch.from_numpy(g).to(s.device)

    # Window definition

    n = torch.arange(-l, l + 1, device=s.device)
    t0 = n / N
    t0 = t0.reshape((-1,))
    a = np.pi / sigma**2
    gp = -2 * a * t0 * g
    gpp = (-2 * a + 4 * a**2 * t0**2) * g

    # Initialization

    STFT = torch.zeros((Nfft, N), device=s.device)
    SST = torch.zeros((Nfft, N), device=s.device)
    VSST = torch.zeros((Nfft, N), device=s.device)

    omega = torch.zeros((Nfft, N), device=s.device)
    tau = torch.zeros((Nfft, N), device=s.device)
    omega2 = torch.zeros((Nfft, N), device=s.device)
    phipp = torch.zeros([Nfft, N], device=s.device, dtype=torch.complex)

    for b in range(N):
        # STFT, window g
        time_inst = torch.arange(
            -min([l, b - 1]), min([l, N - b]), device=s.device
        )
        tmp = torch.fft.fft(s[bt[b] + time_inst] * g[l + time_inst + 1], Nfft)
        vg = tmp[ft]

        # STFT, window xg
        tmp = torch.fft.fft(
            s[bt[b] + time_inst] * (time_inst) / N * g[l + time_inst + 1], Nfft
        )
        vxg = tmp[ft]

        # operator Lx (dtau)

        tau[b, :] = vxg / vg

        # STFT, window gp
        tmp = torch.fft.fft(s[bt[b] + time_inst] * gp[l + time_inst + 1], Nfft)
        vgp = tmp[ft]

        # operator omega
        omega[b, :] = N / Nfft * (ft - 1) - torch.real(
            vgp / 2 / 1j / np.pi / vg
        )

        # STFT, window gpp
        tmp = torch.fft.fft(
            s[bt[b] + time_inst] * gpp[l + time_inst + 1], Nfft
        )
        vgpp = tmp[ft]

        # STFT, windox xgp
        tmp = torch.fft.fft(
            s[bt[b] + time_inst] * (time_inst) / N * gp[l + time_inst + 1],
            Nfft,
        )
        vxgp = tmp[ft]

        # computation of the two different omega

        phipp[b, :] = (
            1 / 2 / 1j / np.pi * (vgpp * vg - vgp**2) / (vxg * vgp - vxgp * vg)
        )
        print(phipp.dtype)

        # new omega2
        omega2[b, :] = (
            omega[b, :]
            - torch.real(phipp[b, :]) * torch.real(tau[b, :])
            + torch.imag(phipp[b, :]) * torch.imag(tau[b, :])
        )

        # Storing STFT
        STFT[b, :] = vg * torch.exp(
            2 * 1j * np.pi * (ft - 1) * min(l, b - 1) / Nfft
        )

    # reassignment step
    for b in range(N):
        for eta in range(Nfft):
            if abs(STFT(eta, b)) > gamma:
                # Original reassignment
                k = 1 + int(round(Nfft / N * omega(eta, b)))
                if (1 <= k) and (k <= Nfft):
                    SST[k, b] += STFT[eta, b]

                # Reassignment using new omega2
                k = 1 + int(round(Nfft / N * omega2(eta, b)))
                if (1 <= k) and (k <= Nfft):
                    VSST[k, b] += STFT[eta, b]
    return STFT, SST, VSST, omega, omega2
