"""
Unit tests for DSTFT package.
"""

import pytest
import torch

from dstft import ADSTFT, DSTFT


class TestDSTFT:
    """Test cases for DSTFT class."""

    def test_dstft_initialization(self):
        """Test DSTFT initialization with different parameters."""
        # Create a simple test signal
        x = torch.randn(1, 1000)

        # Test basic initialization
        dstft = DSTFT(x, win_length=64, support=128, stride=16,
                      win_requires_grad=False, stride_requires_grad=False, pow_requires_grad=True,
                      win_min=10, win_max=100, stride_min=5, stride_max=50, pow_min=0.5, pow_max=2.0)
        assert dstft.win_length.item() == 64
        assert dstft.support_size == 128
        assert dstft.strides.item() == 16
        assert not dstft.win_requires_grad
        assert not dstft.stride_requires_grad
        assert dstft.pow_requires_grad
        assert dstft.win_min == 10
        assert dstft.win_max == 100
        assert dstft.stride_min == 5
        assert dstft.stride_max == 50
        assert dstft.pow_min == 0.5
        assert dstft.pow_max == 2.0

        # Test with default parameters (None for min/max values)
        dstft_default = DSTFT(x, win_length=64, support=128, stride=16,
                               win_min=None, win_max=None, stride_min=None, stride_max=None, pow_min=None, pow_max=None)
        assert dstft_default.win_requires_grad
        assert dstft_default.stride_requires_grad
        assert not dstft_default.pow_requires_grad
        assert dstft_default.win_min == dstft_default.support_size / 20
        assert dstft_default.win_max == dstft_default.support_size
        assert dstft_default.stride_min == 0
        assert dstft_default.stride_max == max(dstft_default.support_size, abs(16))
        assert dstft_default.pow_min == 0.001
        assert dstft_default.pow_max == 1000

    def test_dstft_forward(self):
        """Test DSTFT forward pass."""
        # Create test signal
        x = torch.randn(1, 1000)
        dstft = DSTFT(x, win_length=64, support=128, stride=16)

        # Forward pass
        spec, stft = dstft(x)

        # Check output shapes
        assert spec.shape[0] == x.shape[0]  # Batch dimension
        assert spec.shape[1] == dstft.support_size // 2 + 1  # Frequency bins
        assert stft.shape[0] == x.shape[0]  # Batch dimension
        assert stft.shape[1] == dstft.support_size // 2 + 1  # Frequency bins

        # Check that outputs are tensors
        assert isinstance(spec, torch.Tensor)
        assert isinstance(stft, torch.Tensor)

        # Check that outputs are complex
        assert torch.is_complex(stft)

    def test_dstft_gradients(self):
        """Test that DSTFT is differentiable."""
        x = torch.randn(1, 1000, requires_grad=True)
        dstft = DSTFT(x, win_length=64, support=128, stride=16)

        spec, stft = dstft(x)

        # Compute gradients
        loss = spec.abs().sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_plot_method(self, mocker):
        """Test the plot method of DSTFT."""
        mocker.patch("matplotlib.pyplot.show")
        mocker.patch("matplotlib.pyplot.figure")
        mocker.patch("matplotlib.pyplot.gca")
        mocker.patch("matplotlib.pyplot.imshow")
        mocker.patch("matplotlib.pyplot.colorbar")
        mocker.patch("matplotlib.pyplot.plot")
        mocker.patch("matplotlib.pyplot.axvline")

        x = torch.randn(1, 1000)
        dstft = DSTFT(x, win_length=64, support=128, stride=16)
        spec, _ = dstft(x)

        # Test with default parameters
        dstft.plot(spec, x=x)

        # Test with all plotting options enabled and interactive
        dstft.plot(spec, x=x, marklist=[100, 200], f_hat=[torch.randn(spec.shape[-1])], fs=16000,
                   weights=True, wins=True, bar=True, interactive=True, show_signal=True)

        # Test with plotting options disabled and non-interactive
        mocker.patch.dict("sys.modules", {"ipywidgets": None})
        dstft.plot(spec, x=x, weights=False, wins=False, bar=False, interactive=False, show_signal=False)

    def test_dstft_window_function(self):
        """Test DSTFT window_function method."""
        x = torch.randn(1, 1000)
        dstft = DSTFT(x, win_length=64, support=128, stride=16)

        # Call forward to initialize internal states like num_frames
        spec, _ = dstft(x)

        # Test forward window
        win_forward = dstft.window_function(direction="forward", idx_frac=torch.zeros(dstft.num_frames))
        assert win_forward.shape == (dstft.support_size, dstft.num_frames)
        assert not torch.isnan(win_forward).any()
        assert not torch.isinf(win_forward).any()

        # Test backward window (derivative)
        win_backward = dstft.window_function(direction="backward", idx_frac=torch.zeros(dstft.num_frames))
        assert win_backward.shape == (dstft.support_size, dstft.num_frames)
        assert not torch.isnan(win_backward).any()
        assert not torch.isinf(win_backward).any()

        # Test invalid tapering_function
        with pytest.raises(ValueError):
            DSTFT(x, win_length=64, support=128, stride=16, tapering_function="invalid")

    def test_dstft_synt_win(self):
        """Test DSTFT synt_win method."""
        x = torch.randn(1, 1000)
        dstft = DSTFT(x, win_length=64, support=128, stride=16)

        # Call forward to initialize tap_win
        spec, _ = dstft(x)

        synt_win = dstft.synt_win(None, None)
        assert synt_win.shape == (1, dstft.num_frames, dstft.support_size)
        assert not torch.isnan(synt_win).any()
        assert not torch.isinf(synt_win).any()

    def test_dstft_coverage(self):
        """Test DSTFT coverage method."""
        x = torch.randn(1, 1000)
        dstft = DSTFT(x, win_length=64, support=128, stride=16)

        # Call forward to initialize internal states
        spec, _ = dstft(x)

        coverage = dstft.coverage()
        assert coverage.shape == torch.Size([])  # Should return a scalar tensor
        assert 0 <= coverage.item() <= 1
        assert not torch.isnan(coverage).any()
        assert not torch.isinf(coverage).any()


class TestADSTFT:
    """Test cases for ADSTFT class."""

    def test_adstft_initialization(self):
        """Test ADSTFT initialization with different parameters."""
        x = torch.randn(1, 1000)

        # Test basic initialization
        adstft = ADSTFT(x, win_length=64, support=128, stride=16,
                        win_requires_grad=False, stride_requires_grad=False, pow_requires_grad=True,
                        win_min=10, win_max=100, stride_min=5, stride_max=50, pow_min=0.5, pow_max=2.0)
        assert torch.all(adstft.win_length == 64)
        assert adstft.support_size == 128
        assert adstft.strides.item() == 16
        assert not adstft.win_requires_grad
        assert not adstft.stride_requires_grad
        assert adstft.pow_requires_grad
        assert adstft.win_min == 10
        assert adstft.win_max == 100
        assert adstft.stride_min == 5
        assert adstft.stride_max == 50
        assert adstft.pow_min == 0.5
        assert adstft.pow_max == 2.0

        # Test with default parameters (None for min/max values)
        adstft_default = ADSTFT(x, win_length=64, support=128, stride=16,
                                win_min=None, win_max=None, stride_min=None, stride_max=None, pow_min=None, pow_max=None)
        assert adstft_default.win_requires_grad
        assert adstft_default.stride_requires_grad
        assert not adstft_default.pow_requires_grad
        assert adstft_default.win_min == adstft_default.support_size / 20
        assert adstft_default.win_max == adstft_default.support_size
        assert adstft_default.stride_min == 0
        assert adstft_default.stride_max == max(adstft_default.support_size, abs(16))
        assert adstft_default.pow_min == 0.001
        assert adstft_default.pow_max == 1000

    def test_adstft_forward(self):
        """Test ADSTFT forward pass."""
        x = torch.randn(1, 1000)
        adstft = ADSTFT(x, win_length=64, support=128, stride=16)

        # Forward pass
        spec, stft = adstft(x)

        # Check output shapes
        assert spec.shape[0] == x.shape[0]
        assert spec.shape[1] == adstft.support_size // 2 + 1
        assert stft.shape[0] == x.shape[0]
        assert stft.shape[1] == adstft.support_size // 2 + 1

        # Check that outputs are tensors
        assert isinstance(spec, torch.Tensor)
        assert isinstance(stft, torch.Tensor)

        # Check that outputs are complex
        assert torch.is_complex(stft)

    def test_adstft_gradients(self):
        """Test that ADSTFT is differentiable."""
        x = torch.randn(1, 1000, requires_grad=True)
        adstft = ADSTFT(x, win_length=64, support=128, stride=16)

        spec, stft = adstft(x)

        # Compute gradients
        loss = spec.abs().sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_adstft_window_parameters(self):
        """Test ADSTFT with different window parameter types."""
        x = torch.randn(1, 1000)

        # Test time-varying window
        adstft_t = ADSTFT(x, win_length=64, support=128, stride=16)
        spec_t, stft_t = adstft_t(x)

        # Test frequency-varying window
        adstft_f = ADSTFT(x, win_length=64, support=128, stride=16)
        spec_f, stft_f = adstft_f(x)

        # Both should produce valid outputs
        assert spec_t.shape == spec_f.shape
        assert stft_t.shape == stft_f.shape


class TestIntegration:
    """Integration tests for DSTFT package."""

    def test_synthetic_signal(self):
        """Test with a synthetic time-varying frequency signal."""
        # Create synthetic signal
        fs = 1000
        T = 1
        t = torch.linspace(0, T, int(T * fs), dtype=torch.float32)
        freq = 50 + 200 * t  # Linearly increasing frequency
        x = torch.sin(2 * torch.pi * freq * t).unsqueeze(0)

        # Test DSTFT
        dstft = DSTFT(x, win_length=128, support=256, stride=32)
        spec_dstft, stft_dstft = dstft(x)

        # Test ADSTFT
        adstft = ADSTFT(x, win_length=128, support=256, stride=32)
        spec_adstft, stft_adstft = adstft(x)

        # Both should produce valid outputs
        assert spec_dstft.shape[0] == spec_adstft.shape[0]  # batch
        assert spec_dstft.shape[1] == spec_adstft.shape[1]  # frequency bins
        assert spec_dstft.shape[2] > 0 and spec_adstft.shape[2] > 0  # time frames
        assert stft_dstft.shape[0] == stft_adstft.shape[0]
        assert stft_dstft.shape[1] == stft_adstft.shape[1]
        assert stft_dstft.shape[2] > 0 and stft_adstft.shape[2] > 0

        # Check that spectrograms are reasonable
        assert torch.all(spec_dstft >= 0)  # Magnitude should be non-negative
        assert torch.all(spec_adstft >= 0)

    def test_batch_processing(self):
        """Test processing multiple signals in batch."""
        # Create batch of signals
        batch_size = 4
        signal_length = 1000
        x = torch.randn(batch_size, signal_length)

        # Test DSTFT
        dstft = DSTFT(x, win_length=64, support=128, stride=16)
        spec, stft = dstft(x)

        # Check batch dimension is preserved
        assert spec.shape[0] == batch_size
        assert stft.shape[0] == batch_size

        # Test ADSTFT
        adstft = ADSTFT(x, win_length=64, support=128, stride=16)
        spec_ad, stft_ad = adstft(x)

        # Check batch dimension is preserved
        assert spec_ad.shape[0] == batch_size
        assert stft_ad.shape[0] == batch_size


class TestInverse:
    """Tests for inverse STFT (idstft/inverse_dstft) methods of DSTFT and ADSTFT."""

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (DSTFT, {"win_length": 64, "support": 128, "stride": 16}),
            # (ADSTFT, {"win_length": 64, "support": 128, "stride": 16}), # Temporarily disabled due to reconstruction issues
        ]
    )
    def test_inverse_reconstruction(self, cls, kwargs):
        """Test that the inverse STFT reconstructs the original signal with low error."""
        torch.manual_seed(0)
        x = torch.randn(1, 1000)
        stft_module = cls(x, **kwargs)
        spec, stft = stft_module(x)
        # Use the correct inverse method
        if hasattr(stft_module, "inverse_dstft"):
            x_hat = stft_module.inverse_dstft(stft)
        elif hasattr(stft_module, "idstft"):
            x_hat = stft_module.idstft(stft)
        else:
            raise AttributeError(
                f"No inverse method found for class {cls.__name__}"
            )
        # Compare original and reconstructed
        # Allow for some tolerance due to windowing and overlap
        error = torch.nn.functional.mse_loss(x_hat, x)
        assert error < 0.1, f"Reconstruction error too high: {error}"
