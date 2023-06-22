from math import pi

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class DSTFT(nn.Module):
    def __init__(self, x,
                    win_length: float,   
                    support: int,                
                    stride: int, 
                    pow: float = 1.0, 
                    win_pow: float = 1.0,
                    win_p: str = None,
                    stride_p: str = None,
                    pow_p: str = None,       
                    win_requires_grad: bool = True,
                    stride_requires_grad: bool = True,
                    pow_requires_grad: bool = False,             
                    #params: str = 'p_tf', # p, p_t, p_f, p_tf
                    win_min = None,
                    win_max = None,
                    stride_min = None,
                    stride_max = None,    
                    
                    tapering_function: str = 'hann',
                    sr : int = 16_000,
                    window_transform = None,
                    stride_transform = None,
                    dynamic_parameter : bool = False,
                    first_frame : bool = False,
                    ):
        super(DSTFT, self).__init__()     
        
        # Constants and hyperparameters
        self.N = support                        # support size
        self.F = int(1 + self.N/2)              # nb of frequencies
        self.B = x.shape[0]                     # batch size
        self.L = x.shape[-1]                    # signal length
        self.device = x.device
        self.dtype = x.dtype
        
        self.win_requires_grad = win_requires_grad
        self.stride_requires_grad = stride_requires_grad
        self.pow_requires_grad = pow_requires_grad
        self.tapering_function = tapering_function
        self.dynamic_parameter = dynamic_parameter
        self.first_frame = first_frame
        self.sr = sr
        self.pow = pow
        self.tap_win = None
        
        # Register eps and min as a buffer tensor
        self.register_buffer('eps', torch.tensor(torch.finfo(torch.float).eps, dtype=self.dtype, device=self.device))
        self.register_buffer('min', torch.tensor(torch.finfo(torch.float).min, dtype=self.dtype, device=self.device))
        
        # Calculate the number of frames
        self.T = int(1 + torch.div(x.shape[-1] - (self.N - 1) - 1, stride, rounding_mode='floor'))        
        #self.T = int(1 + torch.div(self.L - self.N/2 , stride, rounding_mode='floor'))
        
        if win_min is None: self.win_min = self.N / 20 # 0
        else: self.win_min = win_min        
        if win_max is None: self.win_max = self.N
        else: self.win_max = win_max
        if stride_min is None: self.stride_min = 0
        else: self.stride_min = stride_min        
        if stride_max is None: self.stride_max = max(self.N, abs(stride))
        else: self.stride_max = stride_max
        
        # HOP LENGTH / FRAME INDEX
        if stride_transform is None: self.stride_transform = self.__stride_transform
        else: self.stride_transform = stride_transform
        # Determine the shape of the stride/hop-length/ frame index parameters
        if   stride_p is None: stride_size = (1,)
        elif stride_p == 't':  stride_size = (self.T,)   
        else: print('stride_p error', stride_p)
        # Create the window length parameter and assign it the appropriate shape
        self.strides = nn.Parameter(torch.full(stride_size, abs(stride), dtype=self.dtype, device=self.device), requires_grad=self.stride_requires_grad)     
        #print(self.strides)
        #self.stride = stride       
        
        # WIN LENGTH
        # win length constraints
        if window_transform is None: self.window_transform = self.__window_transform
        else: self.window_transform = window_transform
        # Determine the shape of the window length parameters
        if   win_p is None: win_length_size = (1, 1)
        elif win_p == 't':  win_length_size = (1, self.T)
        elif win_p == 'f':  win_length_size = (self.N, 1)
        elif win_p == 'tf': win_length_size = (self.N, self.T)
        else: print('win_p error', win_p)
        # Create the window length parameter and assign it the appropriate shape
        self.win_length = nn.Parameter(torch.full(win_length_size, abs(win_length), dtype=self.dtype, device=self.device), requires_grad=self.win_requires_grad)
        
        # WIN POW
        if   pow_p is None: win_pow_size = (1, 1)
        elif pow_p == 't':  win_pow_size = (1, self.T)
        elif pow_p == 'f':  win_pow_size = (self.N, 1)
        elif pow_p == 'tf': win_pow_size = (self.N, self.T)
        else: print('pow_p error', pow_p)
        self.win_pow = nn.Parameter(torch.full(win_pow_size, abs(win_pow), dtype=self.dtype, device=self.device), requires_grad=self.pow_requires_grad)


    def __window_transform(self, w_in): #
        w_out = torch.minimum(torch.maximum(w_in, torch.full_like(w_in, self.win_min, dtype=self.dtype, device=self.device)), torch.full_like(w_in, self.win_max, dtype=self.dtype, device=self.device))
        return w_out
    
    def __stride_transform(self, s_in): # born stride entre 0 et 2N 
        s_out = torch.minimum(torch.maximum(s_in, torch.full_like(s_in, self.stride_min, dtype=self.dtype, device=self.device)), torch.full_like(s_in, self.stride_max, dtype=self.dtype, device=self.device))
        return s_out 
    
    @property 
    def actual_win_length(self): # contraints
        return self.window_transform(self.win_length)
    
    @property 
    def actual_strides(self): # stride contraints, actual stride between frames
        return self.stride_transform(self.strides)
    
    @property
    def frames(self):
        # Compute the temporal position (indices) of frames (support)
        expanded_stride = self.actual_strides.expand((self.T,))    
        frames = torch.zeros_like(expanded_stride)
        if self.first_frame:
            frames[0] = (self.actual_win_length.expand((self.N, self.T))[:, 0].max(dim=0, keepdim=False)[0] - self.N)/2
        frames[1:] = frames[0] + expanded_stride[1:].cumsum(dim=0)        
        
        #frames = torch.cumsum(self.actual_strides, dim=0)
        #print(frames)
        return frames
    
    @property 
    def effective_strides(self):
        # Compute the strides between window (and not frames)
        expanded_stride = self.actual_strides.expand((self.T,))    
        effective_strides = torch.zeros_like(expanded_stride)
        effective_strides[1:] = expanded_stride[1:]
        cat = torch.cat((torch.tensor([self.N], dtype=self.dtype, device=self.device), self.actual_win_length.expand((self.N, self.T)).max(dim=0, keepdim=False)[0]), dim=0).diff()/2        
        effective_strides = effective_strides - cat
        return effective_strides
    
    def forward(self, x):
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, 'forward')
        real, imag, spec, phase = stft.real, stft.imag, stft.abs().pow(self.pow)[:, :self.F], stft.angle()[:, :self.F]
        spec = spec + torch.finfo(x.dtype).eps
        return spec, stft, real, imag, phase
    
    def backward(self, x, dl_ds):
        # Compute the gradient of the loss w.r.t. window length parameter with the chain rule
        dstft_dp = self.stft(x, 'backward')
        dl_dp = (torch.conj(dl_ds) * dstft_dp).sum().real #dl_dp = (dl_ds.real * dstft_dp.real + dl_ds.imag * dstft_dp.imag).sum()        
        return dl_dp.unsqueeze(0)
    
    def stft(self, x: torch.tensor, direction: str):
        #batch_size, length, device, dtype = x.shape[0], x.shape[-1], x.device, x.dtype     
        
        # Generate strided signal and shift idx_frac
        strided_x, idx_frac = self.stride(x) # B, T, N; T

        # Generate the tapering window function for the STFT
        self.tap_win = self.window_function(direction=direction, idx_frac=idx_frac).permute(2, 1, 0)  # T, N, N
        
        # Generate tapering function shift   
        shift = torch.arange(end=self.N, device=self.device, dtype=self.dtype, requires_grad=False)
        shift = idx_frac[:, None] * shift[None, :] # T, N
        
        # Compute tapered x
        strided_x = strided_x[:, :, None, :] # B, T, 1, N
        self.tap_win = self.tap_win[None, :, :, :] # 1, T, N, 1
        shift = torch.exp( 2j * pi * shift / self.N)[None, :, :, None] # 1, T, N, 1
        tapered_x = strided_x * self.tap_win * shift # B, T, N, N
        
        # Generate Fourier coefficients    
        coeff = torch.arange(end=self.N, device=self.device, dtype=self.dtype, requires_grad=False) 
        coeff = coeff[:, None] @ coeff[None, :]
        coeff = torch.exp(- 2j * pi * coeff / self.N) # N, N
        
        # Perform the STFT
        coeff = coeff[None, None, :, :]  # 1, 1, N, N     
        stft = (tapered_x * coeff).sum(dim=-1)  # B, T, N        #stft = torch.einsum('...ij,...jk->...ik', tapered_x, coeff)
        stft = stft.permute(0, 2, 1)         # B, N, T           #stft = stft.transpose(-1, -2)  

            
        return stft
    
    def stride(self, x) -> torch.tensor:                
        # frames index and strided x
        idx_floor = self.frames.floor()
        idx_frac = self.frames - idx_floor
        idx_floor = idx_floor.long()[:, None].expand((self.T, self.N)) + torch.arange(0, self.N, device=self.device)
        idx_floor[idx_floor >= self.L] = -1
        strided_x = x[:, idx_floor]
        strided_x[:, idx_floor < 0] = 0
        return strided_x, idx_frac
    
    def window_function(self, direction: str, idx_frac) -> torch.tensor:
        if self.tapering_function not in {'hann', 'hanning',}:
            raise ValueError(f"tapering_function must be one of '{'hann', 'hanning',}', but got padding_mode='{self.tapering_function}'")
        else:
            # Create an array of indices to use as the base for the window function
            base = torch.arange(0, self.N, 1, dtype=self.dtype, device=self.device)[:, None, None].expand([-1, self.N, self.T])   
            base = base - idx_frac
            # Expand the win_length parameter to match the shape of the base array         
            #if self.actual_win_length.dim() == 3:
            #    self.expanded_win_length = self.actual_win_length.expand([self.N, self.N, self.T])
            #elif self.actual_win_length.dim() == 1:
            #    self.expanded_win_length = self.actual_win_length[:, None, None].expand([self.N, self.N, self.T])
            #elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.T:
            #    self.expanded_win_length = self.actual_win_length[:, None, :].expand([self.N, self.N, self.T])
            #elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.N: 
            #    self.expanded_win_length = self.actual_win_length[:, :, None].expand([self.N, self.N, self.T])          
        
        # calculate the tapering function and its derivate w.r.t. window length
        if self.tapering_function == 'hann' or self.tapering_function == 'hanning':
            if direction == 'forward':
                self.tap_win = 0.5 - 0.5 * torch.cos(2 * pi * (base + (self.actual_win_length-self.N+1)/2) / self.actual_win_length )                
                mask1 = base.ge(torch.ceil( (self.N-1+self.actual_win_length)/2))
                mask2 = base.le(torch.floor((self.N-1-self.actual_win_length)/2))            
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                self.tap_win = self.tap_win / self.tap_win.sum(dim=0, keepdim=True) 
                return self.tap_win.pow(self.win_pow)
            
            elif direction == 'backward':
                f = torch.sin(2 * pi * (base - (self.N-1)/2) / self.actual_win_length)            
                d_tap_win = - pi / self.actual_win_length * ((self.N-1)/2 - base) * f
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                return d_tap_win
            
    def coverage(self): # in [0, 1]    
        # compute coverage
        expanded_win, _ = self.actual_win_length.expand((self.N, self.T)).min(dim=0, keepdim=False)
        cov = expanded_win[0]
        maxi = self.frames[0] + self.N/2 + expanded_win[0]/2        
        for i in range(1, self.T):
            start = torch.min(self.L*torch.ones_like(expanded_win[i]), torch.max(torch.zeros_like(expanded_win[i]), self.frames[i] + self.N/2 - expanded_win[i]/2))
            end = torch.min(self.L*torch.ones_like(expanded_win[i]), torch.max(torch.zeros_like(expanded_win[i]), self.frames[i] + self.N/2 + expanded_win[i]/2))
            if end > maxi:
                cov += end - torch.max(start, maxi)
                maxi = end
        cov /= self.L
        return cov

    def print(self, spec, x=None, marklist=None, weights=True, wins=True, bar=False):
        plt.figure()
        plt.title('Spectrogram')
        ax = plt.subplot()
        im = ax.imshow(spec[0].detach().cpu().log(), aspect='auto', origin='lower', cmap='jet', extent=[0,spec.shape[-1], 0, spec.shape[-2]])
        plt.ylabel('frequencies')
        plt.xlabel('frames')
        if bar == True: plt.colorbar(im, ax=ax)
        plt.show()

        if weights == True:
            plt.figure()
            plt.title('Distribution of window lengths')
            ax = plt.subplot()
            im = ax.imshow(self.actual_win_length[:self.F].detach().cpu(), aspect='auto', origin='lower', cmap='jet')
            ax.set_ylabel('frequencies')
            ax.set_xlabel('frames')
            if bar == True : 
                plt.colorbar(im, ax=ax)
                im.set_clim(self.win_min, self.win_max)   
            plt.show()   

        if self.tap_win is not None and wins == True:
            fig, ax = plt.subplots()
            ax.plot(self.T + .5 + x.squeeze().cpu().numpy(), linewidth=1,)
            for i, start in enumerate(self.frames.detach().cpu()):
                ax.plot(range(int(start.floor().item()), int(start.floor().item()+self.N)), self.T-i-1.3 + 150 * self.tap_win[:, i, :, :].mean(dim=1).squeeze().detach().cpu(), c='#1f77b4')

            if marklist is not None:
                for elem in marklist:
                    plt.axvline(elem, 0, self.T, c='gray')
            else:
                ax.axvline(x=0, ymin=0, ymax=self.T, c='gray')
                ax.axvline(x=x.shape[-1], ymin=0, ymax=self.T, c='gray')
            plt.show()






class FDSTFT(nn.Module):
    def __init__(self, x,
                    win_length: float,   
                    support: int,                
                    stride: int, 
                    pow: float = 1.0, 
                    win_pow: float = 1.0,
                    win_p: str = None,
                    stride_p: str = None,
                    pow_p: str = None,       
                    win_requires_grad: bool = True,
                    stride_requires_grad: bool = True,
                    pow_requires_grad: bool = False,             
                    #params: str = 'p_tf', # p, p_t, p_f, p_tf
                    win_min = None,
                    win_max = None,
                    stride_min = None,
                    stride_max = None,    
                    
                    tapering_function: str = 'hann',
                    sr : int = 16_000,
                    window_transform = None,
                    stride_transform = None,
                    dynamic_parameter : bool = False,
                    first_frame : bool = False,
                    ):
        super(FDSTFT, self).__init__()     
        
        # Constants and hyperparameters
        self.N = support                        # support size
        self.F = int(1 + self.N/2)              # nb of frequencies
        self.B = x.shape[0]                     # batch size
        self.L = x.shape[-1]                    # signal length
        self.device = x.device
        self.dtype = x.dtype
        
        self.win_requires_grad = win_requires_grad
        self.stride_requires_grad = stride_requires_grad
        self.pow_requires_grad = pow_requires_grad
        self.tapering_function = tapering_function
        self.dynamic_parameter = dynamic_parameter
        self.first_frame = first_frame
        self.sr = sr
        self.pow = pow
        self.tap_win = None
        
        # Register eps and min as a buffer tensor
        self.register_buffer('eps', torch.tensor(torch.finfo(torch.float).eps, dtype=self.dtype, device=self.device))
        self.register_buffer('min', torch.tensor(torch.finfo(torch.float).min, dtype=self.dtype, device=self.device))
        
        # Calculate the number of frames
        self.T = int(1 + torch.div(x.shape[-1] - (self.N - 1) - 1, stride, rounding_mode='floor'))        
        #self.T = int(1 + torch.div(self.L - self.N/2 , stride, rounding_mode='floor'))
        
        if win_min is None: self.win_min = self.N / 20 # 0
        else: self.win_min = win_min        
        if win_max is None: self.win_max = self.N
        else: self.win_max = win_max
        if stride_min is None: self.stride_min = 0
        else: self.stride_min = stride_min        
        if stride_max is None: self.stride_max = max(self.N, abs(stride))
        else: self.stride_max = stride_max
        
        # HOP LENGTH / FRAME INDEX
        if stride_transform is None: self.stride_transform = self.__stride_transform
        else: self.stride_transform = stride_transform
        # Determine the shape of the stride/hop-length/ frame index parameters
        if   stride_p is None: stride_size = (1,)
        elif stride_p == 't':  stride_size = (self.T,)   
        else: print('stride_p error', stride_p)
        # Create the window length parameter and assign it the appropriate shape
        self.strides = nn.Parameter(torch.full(stride_size, abs(stride), dtype=self.dtype, device=self.device), requires_grad=self.stride_requires_grad)     
        #print(self.strides)
        #self.stride = stride       
        
        # WIN LENGTH
        # win length constraints
        if window_transform is None: self.window_transform = self.__window_transform
        else: self.window_transform = window_transform
        # Determine the shape of the window length parameters
        if   win_p is None: win_length_size = (1, 1)
        elif win_p == 't':  win_length_size = (1, self.T)
        
        else: print('win_p error', win_p)
        # Create the window length parameter and assign it the appropriate shape
        self.win_length = nn.Parameter(torch.full(win_length_size, abs(win_length), dtype=self.dtype, device=self.device), requires_grad=self.win_requires_grad)
        
        # WIN POW
        if   pow_p is None: win_pow_size = (1, 1)
        elif pow_p == 't':  win_pow_size = (1, self.T)
        else: print('pow_p error', pow_p)
        self.win_pow = nn.Parameter(torch.full(win_pow_size, abs(win_pow), dtype=self.dtype, device=self.device), requires_grad=self.pow_requires_grad)


    def __window_transform(self, w_in): #
        w_out = torch.minimum(torch.maximum(w_in, torch.full_like(w_in, self.win_min, dtype=self.dtype, device=self.device)), torch.full_like(w_in, self.win_max, dtype=self.dtype, device=self.device))
        return w_out
    
    def __stride_transform(self, s_in): # born stride entre 0 et 2N 
        s_out = torch.minimum(torch.maximum(s_in, torch.full_like(s_in, self.stride_min, dtype=self.dtype, device=self.device)), torch.full_like(s_in, self.stride_max, dtype=self.dtype, device=self.device))
        return s_out 
    
    @property 
    def actual_win_length(self): # contraints
        return self.window_transform(self.win_length)
    
    @property 
    def actual_strides(self): # stride contraints, actual stride between frames
        return self.stride_transform(self.strides)
    
    @property
    def frames(self):
        # Compute the temporal position (indices) of frames (support)
        expanded_stride = self.actual_strides.expand((self.T,))    
        frames = torch.zeros_like(expanded_stride)
        if self.first_frame:
            frames[0] = (self.actual_win_length.expand((self.N, self.T))[:, 0].max(dim=0, keepdim=False)[0] - self.N)/2
        frames[1:] = frames[0] + expanded_stride[1:].cumsum(dim=0)        
        
        #frames = torch.cumsum(self.actual_strides, dim=0)
        #print(frames)
        return frames
    
    @property 
    def effective_strides(self):
        # Compute the strides between window (and not frames)
        expanded_stride = self.actual_strides.expand((self.T,))    
        effective_strides = torch.zeros_like(expanded_stride)
        effective_strides[1:] = expanded_stride[1:]
        cat = torch.cat((torch.tensor([self.N], dtype=self.dtype, device=self.device), self.actual_win_length.expand((self.N, self.T)).max(dim=0, keepdim=False)[0]), dim=0).diff()/2        
        effective_strides = effective_strides - cat
        return effective_strides
    
    def forward(self, x):
        # Perform the forward STFT and extract the magnitude, phase, real, and imaginary parts
        stft = self.stft(x, 'forward')
        real, imag, spec, phase = stft.real, stft.imag, stft.abs().pow(self.pow)[:, :self.F], stft.angle()[:, :self.F]
        spec = spec + torch.finfo(x.dtype).eps
        return spec, stft, real, imag, phase
    
    def backward(self, x, dl_ds):
        # Compute the gradient of the loss w.r.t. window length parameter with the chain rule
        dstft_dp = self.stft(x, 'backward')
        dl_dp = (torch.conj(dl_ds) * dstft_dp).sum().real #dl_dp = (dl_ds.real * dstft_dp.real + dl_ds.imag * dstft_dp.imag).sum()        
        return dl_dp.unsqueeze(0)
    
    def stft(self, x: torch.tensor, direction: str):
        #batch_size, length, device, dtype = x.shape[0], x.shape[-1], x.device, x.dtype     
        
        # Generate strided signal and shift idx_frac
        strided_x, idx_frac = self.stride(x) # B, T, N; T

        # Generate the tapering window function for the STFT
        self.tap_win = self.window_function(direction=direction, idx_frac=idx_frac).permute(1, 0)  # T, N
        
        
        
        # Compute tapered x
        strided_x = strided_x[:, :, :] # B, T, N
        self.tap_win = self.tap_win[None, :, :] # 1, T, 1
        
        tapered_x = strided_x * self.tap_win #* shift # B, T, N, N
        spectr = torch.fft.fft(tapered_x)
        shift = torch.arange(end=self.N, device=self.device, dtype=self.dtype, requires_grad=False)
        shift = (idx_frac[:, None] * shift[None, :]) # T, N
        shift = torch.exp( 2j * pi * shift / self.N)[None, ...] # 1, T, N
        print(shift)
        
        stft = spectr #* shift
        print(stft.shape)

            
        return stft.permute(0, 2, 1)
    
    def stride(self, x) -> torch.tensor:                
        # frames index and strided x
        idx_floor = self.frames.floor()
        idx_frac = self.frames - idx_floor
        idx_floor = idx_floor.long()[:, None].expand((self.T, self.N)) + torch.arange(0, self.N, device=self.device)
        idx_floor[idx_floor >= self.L] = -1
        strided_x = x[:, idx_floor]
        strided_x[:, idx_floor < 0] = 0
        return strided_x, idx_frac
    
    def window_function(self, direction: str, idx_frac) -> torch.tensor:
        if self.tapering_function not in {'hann', 'hanning',}:
            raise ValueError(f"tapering_function must be one of '{'hann', 'hanning',}', but got padding_mode='{self.tapering_function}'")
        else:
            # Create an array of indices to use as the base for the window function
            base = torch.arange(0, self.N, 1, dtype=self.dtype, device=self.device)[:, None].expand([-1, self.T])   
            base = base - idx_frac
            # Expand the win_length parameter to match the shape of the base array         
            #if self.actual_win_length.dim() == 3:
            #    self.expanded_win_length = self.actual_win_length.expand([self.N, self.N, self.T])
            #elif self.actual_win_length.dim() == 1:
            #    self.expanded_win_length = self.actual_win_length[:, None, None].expand([self.N, self.N, self.T])
            #elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.T:
            #    self.expanded_win_length = self.actual_win_length[:, None, :].expand([self.N, self.N, self.T])
            #elif self.actual_win_length.dim() == 2 and self.actual_win_length.shape[-1] == self.N: 
            #    self.expanded_win_length = self.actual_win_length[:, :, None].expand([self.N, self.N, self.T])          
        
        # calculate the tapering function and its derivate w.r.t. window length
        if self.tapering_function == 'hann' or self.tapering_function == 'hanning':
            if direction == 'forward':
                self.tap_win = 0.5 - 0.5 * torch.cos(2 * pi * (base + (self.actual_win_length-self.N+1)/2) / self.actual_win_length )                
                mask1 = base.ge(torch.ceil( (self.N-1+self.actual_win_length)/2))
                mask2 = base.le(torch.floor((self.N-1-self.actual_win_length)/2))            
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                self.tap_win = self.tap_win / self.tap_win.sum(dim=0, keepdim=True) 
                return self.tap_win.pow(self.win_pow)
            
            elif direction == 'backward':
                f = torch.sin(2 * pi * (base - (self.N-1)/2) / self.actual_win_length)            
                d_tap_win = - pi / self.actual_win_length * ((self.N-1)/2 - base) * f
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                return d_tap_win
            
    def coverage(self): # in [0, 1]    
        # compute coverage
        expanded_win, _ = self.actual_win_length.expand((self.N, self.T)).min(dim=0, keepdim=False)
        cov = expanded_win[0]
        maxi = self.frames[0] + self.N/2 + expanded_win[0]/2        
        for i in range(1, self.T):
            start = torch.min(self.L*torch.ones_like(expanded_win[i]), torch.max(torch.zeros_like(expanded_win[i]), self.frames[i] + self.N/2 - expanded_win[i]/2))
            end = torch.min(self.L*torch.ones_like(expanded_win[i]), torch.max(torch.zeros_like(expanded_win[i]), self.frames[i] + self.N/2 + expanded_win[i]/2))
            if end > maxi:
                cov += end - torch.max(start, maxi)
                maxi = end
        cov /= self.L
        return cov

    def print(self, spec, x=None, marklist=None, weights=True, wins=True, bar=False):
        plt.figure()
        plt.title('Spectrogram')
        ax = plt.subplot()
        im = ax.imshow(spec[0].detach().cpu().log(), aspect='auto', origin='lower', cmap='jet', extent=[0,spec.shape[-1], 0, spec.shape[-2]])
        plt.ylabel('frequencies')
        plt.xlabel('frames')
        if bar == True: plt.colorbar(im, ax=ax)
        plt.show()

        if weights == True:
            plt.figure()
            plt.title('Distribution of window lengths')
            ax = plt.subplot()
            im = ax.imshow(self.actual_win_length[:self.F].detach().cpu(), aspect='auto', origin='lower', cmap='jet')
            ax.set_ylabel('frequencies')
            ax.set_xlabel('frames')
            if bar == True : 
                plt.colorbar(im, ax=ax)
                im.set_clim(self.win_min, self.win_max)   
            plt.show()   

        if self.tap_win is not None and wins == True:
            fig, ax = plt.subplots()
            ax.plot(self.T + .5 + x.squeeze().cpu().numpy(), linewidth=1,)
            for i, start in enumerate(self.frames.detach().cpu()):
                ax.plot(range(int(start.floor().item()), int(start.floor().item()+self.N)), self.T-i-1.3 + 150 * self.tap_win[:, i, :, :].mean(dim=1).squeeze().detach().cpu(), c='#1f77b4')

            if marklist is not None:
                for elem in marklist:
                    plt.axvline(elem, 0, self.T, c='gray')
            else:
                ax.axvline(x=0, ymin=0, ymax=self.T, c='gray')
                ax.axvline(x=x.shape[-1], ymin=0, ymax=self.T, c='gray')
            plt.show()
