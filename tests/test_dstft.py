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
        dstft = DSTFT(x, win_length=64, support=128, stride=16)
        assert dstft.win_length == 64
        assert dstft.support == 128
        assert dstft.stride == 16

        # Test with different parameters
        dstft2 = DSTFT(x, win_length=128, support=256, stride=32)
        assert dstft2.win_length == 128
        assert dstft2.support == 256
        assert dstft2.stride == 32

    def test_dstft_forward(self):
        """Test DSTFT forward pass."""
        # Create test signal
        x = torch.randn(1, 1000)
        dstft = DSTFT(x, win_length=64, support=128, stride=16)

        # Forward pass
        spec, stft = dstft(x)

        # Check output shapes
        assert spec.shape[0] == x.shape[0]  # Batch dimension
        assert spec.shape[1] == dstft.support // 2 + 1  # Frequency bins
        assert stft.shape[0] == x.shape[0]  # Batch dimension
        assert stft.shape[1] == dstft.support // 2 + 1  # Frequency bins

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


class TestADSTFT:
    """Test cases for ADSTFT class."""

    def test_adstft_initialization(self):
        """Test ADSTFT initialization with different parameters."""
        x = torch.randn(1, 1000)

        # Test basic initialization
        adstft = ADSTFT(x, win_length=64, support=128, stride=16, win_p="t")
        assert adstft.win_length == 64
        assert adstft.support == 128
        assert adstft.stride == 16
        assert adstft.win_p == "t"

        # Test with different window parameter
        adstft2 = ADSTFT(x, win_length=128, support=256, stride=32, win_p="f")
        assert adstft2.win_p == "f"

    def test_adstft_forward(self):
        """Test ADSTFT forward pass."""
        x = torch.randn(1, 1000)
        adstft = ADSTFT(x, win_length=64, support=128, stride=16, win_p="t")

        # Forward pass
        spec, stft = adstft(x)

        # Check output shapes
        assert spec.shape[0] == x.shape[0]
        assert spec.shape[1] == adstft.support // 2 + 1
        assert stft.shape[0] == x.shape[0]
        assert stft.shape[1] == adstft.support // 2 + 1

        # Check that outputs are tensors
        assert isinstance(spec, torch.Tensor)
        assert isinstance(stft, torch.Tensor)

        # Check that outputs are complex
        assert torch.is_complex(stft)

    def test_adstft_gradients(self):
        """Test that ADSTFT is differentiable."""
        x = torch.randn(1, 1000, requires_grad=True)
        adstft = ADSTFT(x, win_length=64, support=128, stride=16, win_p="t")

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
        adstft_t = ADSTFT(x, win_length=64, support=128, stride=16, win_p="t")
        spec_t, stft_t = adstft_t(x)

        # Test frequency-varying window
        adstft_f = ADSTFT(x, win_length=64, support=128, stride=16, win_p="f")
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
        adstft = ADSTFT(x, win_length=128, support=256, stride=32, win_p="t")
        spec_adstft, stft_adstft = adstft(x)

        # Both should produce valid outputs
        assert spec_dstft.shape == spec_adstft.shape
        assert stft_dstft.shape == stft_adstft.shape

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
        adstft = ADSTFT(x, win_length=64, support=128, stride=16, win_p="t")
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
            (
                ADSTFT,
                {"win_length": 64, "support": 128, "stride": 16, "win_p": "t"},
            ),
        ],
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


if __name__ == "__main__":
    pytest.main([__file__])
