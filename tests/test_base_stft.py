import pytest
import torch
from dstft.base import BaseSTFT

class TestBaseSTFT:
    @pytest.fixture
    def base_stft_instance(self):
        x = torch.randn(1, 1000)
        return BaseSTFT(x, win_length=64, support=128, stride=16)

    def test_clamp_parameter(self, base_stft_instance):
        value = torch.tensor([5.0, 15.0, 25.0], dtype=torch.float32)
        min_val = 10.0
        max_val = 20.0
        clamped_value = base_stft_instance._clamp_parameter(value, min_val, max_val)
        expected_value = torch.tensor([10.0, 15.0, 20.0], dtype=torch.float32)
        assert torch.allclose(clamped_value, expected_value)

    def test_init_default_params(self):
        x = torch.randn(1, 1000)
        support = 128
        stride = 16
        win_length = 64

        # Test with default parameters (None for min/max values and transforms)
        base_stft_default = BaseSTFT(x, win_length=win_length, support=support, stride=stride,
                                     win_min=None, win_max=None, stride_min=None, stride_max=None,
                                     pow_min=None, pow_max=None, window_transform=None, stride_transform=None)

        assert base_stft_default.win_min == support / 20
        assert base_stft_default.win_max == support
        assert base_stft_default.stride_min == 0
        assert base_stft_default.stride_max == max(support, abs(stride))
        assert base_stft_default.pow_min == 0.001
        assert base_stft_default.pow_max == 1000
        assert base_stft_default.window_transform == base_stft_default._window_transform
        assert base_stft_default.stride_transform == base_stft_default._stride_transform

    def test_init_custom_params(self):
        x = torch.randn(1, 1000)
        win_length = 64
        support = 128
        stride = 16

        def custom_win_transform(w): return w * 2
        def custom_stride_transform(s): return s + 10

        base_stft_custom = BaseSTFT(x, win_length=win_length, support=support, stride=stride,
                                    win_min=10, win_max=100, stride_min=5, stride_max=50,
                                    pow_min=0.5, pow_max=2.0, window_transform=custom_win_transform,
                                    stride_transform=custom_stride_transform)

        assert base_stft_custom.win_min == 10
        assert base_stft_custom.win_max == 100
        assert base_stft_custom.stride_min == 5
        assert base_stft_custom.stride_max == 50
        assert base_stft_custom.pow_min == 0.5
        assert base_stft_custom.pow_max == 2.0
        assert base_stft_custom.window_transform == custom_win_transform
        assert base_stft_custom.stride_transform == custom_stride_transform

    def test_actual_properties(self, base_stft_instance):
        # Mock win_length, strides, win_pow for testing properties
        base_stft_instance.win_length = torch.tensor(50.0)
        base_stft_instance.strides = torch.tensor(10.0)
        base_stft_instance.win_pow = torch.tensor(1.5)

        # Test actual_win_length
        assert base_stft_instance.actual_win_length.item() == base_stft_instance._clamp_parameter(torch.tensor(50.0), base_stft_instance.win_min, base_stft_instance.win_max).item()

        # Test actual_strides
        assert base_stft_instance.actual_strides.item() == base_stft_instance._clamp_parameter(torch.tensor(10.0), base_stft_instance.stride_min, base_stft_instance.stride_max).item()

        # Test actual_pow
        assert base_stft_instance.actual_pow.item() == base_stft_instance._clamp_parameter(torch.tensor(1.5), base_stft_instance.pow_min, base_stft_instance.pow_max).item()

    def test_transform_methods(self, base_stft_instance):
        # Test _window_transform
        w_in = torch.tensor(70.0)
        clamped_w = base_stft_instance._window_transform(w_in)
        assert clamped_w.item() == base_stft_instance._clamp_parameter(w_in, base_stft_instance.win_min, base_stft_instance.win_max).item()

        # Test _stride_transform
        s_in = torch.tensor(20.0)
        clamped_s = base_stft_instance._stride_transform(s_in)
        assert clamped_s.item() == base_stft_instance._clamp_parameter(s_in, base_stft_instance.stride_min, base_stft_instance.stride_max).item()

        # Test _pow_transform
        p_in = torch.tensor(0.8)
        clamped_p = base_stft_instance._pow_transform(p_in)
        assert clamped_p.item() == base_stft_instance._clamp_parameter(p_in, base_stft_instance.pow_min, base_stft_instance.pow_max).item()

    def test_unfold(self, base_stft_instance):
        x = torch.randn(1, 1000)
        base_stft_instance.frames = torch.tensor([0.0, 16.0, 32.0], dtype=torch.float32) # Example frames
        base_stft_instance.num_frames = base_stft_instance.frames.shape[0]
        base_stft_instance.signal_length = x.shape[-1]

        folded_x, idx_frac = base_stft_instance.unfold(x)

        assert folded_x.shape == (x.shape[0], base_stft_instance.num_frames, base_stft_instance.support_size)
        assert idx_frac.shape == (base_stft_instance.num_frames,)
        assert torch.all(idx_frac >= 0) and torch.all(idx_frac < 1)

    def test_fold(self, base_stft_instance):
        # Create a dummy folded_x for testing fold
        batch_size = 1
        num_frames = 3
        support_size = base_stft_instance.support_size
        signal_length = 1000

        folded_x = torch.randn(batch_size, num_frames, support_size)
        base_stft_instance.frames = torch.tensor([0.0, 16.0, 32.0], dtype=torch.float32) # Example frames
        base_stft_instance.num_frames = num_frames
        base_stft_instance.signal_length = signal_length

        x_hat = base_stft_instance.fold(folded_x)

        assert x_hat.shape == (batch_size, signal_length)
        assert not torch.isnan(x_hat).any()
        assert not torch.isinf(x_hat).any()

    def test_plot_method(self, base_stft_instance, mocker):
        mocker.patch('matplotlib.pyplot.show')
        mocker.patch('matplotlib.pyplot.figure')
        mocker.patch('matplotlib.pyplot.gca')
        mocker.patch('matplotlib.pyplot.imshow')
        mocker.patch('matplotlib.pyplot.colorbar')
        mocker.patch('matplotlib.pyplot.plot')
        mocker.patch('matplotlib.pyplot.axvline')

        # Mock ipywidgets to simulate it not being available
        mocker.patch.dict('sys.modules', {'ipywidgets': None})

        # Create a dummy spectrogram for plotting
        spec = torch.randn(1, 65, 50)
        x = torch.randn(1, 1000)

        # Set some internal states for plot method to work
        base_stft_instance.tap_win = torch.randn(1, 50, 128) # Example shape
        base_stft_instance.frames = torch.arange(0, 50, dtype=torch.float32)
        base_stft_instance.win_length = torch.tensor(64.0)

        # Test with interactive=False to hit the 'else' branch for plot_spectrogram
        base_stft_instance.plot(spec, x=x, interactive=False, weights=True, wins=True, bar=True, show_signal=True)

        # Test with interactive=True (ipywidgets mocked as None, so it should still hit the else branch)
        base_stft_instance.plot(spec, x=x, interactive=True, weights=True, wins=True, bar=True, show_signal=True)

        # Test with all plotting options disabled
        base_stft_instance.plot(spec, x=x, weights=False, wins=False, bar=False, show_signal=False, interactive=False)
