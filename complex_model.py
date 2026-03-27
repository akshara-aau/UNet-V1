import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    """
    Complex convolution: (W_r + j*W_i)(x_r + j*x_i)
    out_real = W_r(x_r) - W_i(x_i)
    out_imag = W_r(x_i) + W_i(x_r)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    def forward(self, real, imag):
        out_real = self.conv_real(real) - self.conv_imag(imag)
        out_imag = self.conv_real(imag) + self.conv_imag(real)
        return out_real, out_imag


class ComplexConvTranspose2d(nn.Module):
    """Complex transposed convolution for the decoder upsampling."""
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.conv_t_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv_t_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    def forward(self, real, imag):
        out_real = self.conv_t_real(real) - self.conv_t_imag(imag)
        out_imag = self.conv_t_real(imag) + self.conv_t_imag(real)
        return out_real, out_imag


class ComplexBatchNorm2d(nn.Module):
    """Independent BN on real and imaginary channels."""
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)

    def forward(self, real, imag):
        return self.bn_real(real), self.bn_imag(imag)


class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, real, imag):
        return self.act(real), self.act(imag)


class ComplexDoubleConv(nn.Module):
    """Two complex conv blocks: (ComplexConv -> ComplexBN -> ComplexLeakyReLU) x2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ComplexConv2d(in_ch, out_ch)
        self.bn1   = ComplexBatchNorm2d(out_ch)
        self.act1  = ComplexLeakyReLU()
        self.conv2 = ComplexConv2d(out_ch, out_ch)
        self.bn2   = ComplexBatchNorm2d(out_ch)
        self.act2  = ComplexLeakyReLU()

    def forward(self, real, imag):
        r, i = self.act1(*self.bn1(*self.conv1(real, imag)))
        r, i = self.act2(*self.bn2(*self.conv2(r, i)))
        return r, i


class ComplexDown(nn.Module):
    """MaxPool2d (applied separately to r/i) -> ComplexDoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ComplexDoubleConv(in_ch, out_ch)

    def forward(self, real, imag):
        return self.conv(self.pool(real), self.pool(imag))


class ComplexUp(nn.Module):
    """Upsample via ComplexConvTranspose2d, pad if needed, cat skip, then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = ComplexConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ComplexDoubleConv(in_ch, out_ch)

    def forward(self, skip_r, skip_i, up_r, up_i):
        up_r, up_i = self.up(up_r, up_i)

        # Pad upsampled tensor to match skip dimensions (handles odd sizes)
        dY = skip_r.size(2) - up_r.size(2)
        dX = skip_r.size(3) - up_r.size(3)
        pad = [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2]
        up_r = F.pad(up_r, pad)
        up_i = F.pad(up_i, pad)

        r = torch.cat([skip_r, up_r], dim=1)
        i = torch.cat([skip_i, up_i], dim=1)
        return self.conv(r, i)


class DeepComplexUNet(nn.Module):
    """
    Phase-Aware Speech Enhancement with Deep Complex U-Net.
    Outputs a Complex Ratio Mask (cRM) bounded by tanh.

    model_size options:
        'dcu8'  - 4 encoder blocks (~6GB VRAM,  batch 16) -- your original
        'dcu16' - 8 encoder blocks (~16GB VRAM, batch 8)  -- recommended
        'dcu20' - 10 encoder blocks (~28GB VRAM, batch 4) -- paper's best
    """
    CHANNEL_CONFIGS = {
        'dcu8':  [32, 64, 128, 256, 512],
        'dcu16': [32, 64, 64, 128, 128, 256, 256, 512, 512],
        'dcu20': [32, 64, 64, 128, 128, 256, 256, 256, 512, 512, 1024],
    }

    def __init__(self, n_channels=1, model_size='dcu16'):
        super().__init__()
        assert model_size in self.CHANNEL_CONFIGS, \
            f"model_size must be one of {list(self.CHANNEL_CONFIGS.keys())}"

        channels = self.CHANNEL_CONFIGS[model_size]
        self.model_size = model_size

        # Encoder
        self.inc   = ComplexDoubleConv(n_channels, channels[0])
        self.downs = nn.ModuleList([
            ComplexDown(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])

        # Decoder (mirrors encoder in reverse)
        dec_ch = list(reversed(channels))
        self.ups = nn.ModuleList([
            ComplexUp(dec_ch[i], dec_ch[i + 1])
            for i in range(len(dec_ch) - 1)
        ])

        # Output: 1x1 complex conv -> tanh bounded cRM
        self.out_conv = ComplexConv2d(channels[0], n_channels, kernel_size=1, padding=0)

    def forward(self, real, imag):
        # Ensure [B, C, F, T]
        if real.dim() == 3:
            real = real.unsqueeze(1)
            imag = imag.unsqueeze(1)

        # Encoder — save all skip connections
        skips_r, skips_i = [], []
        r, i = self.inc(real, imag)
        skips_r.append(r)
        skips_i.append(i)

        for down in self.downs:
            r, i = down(r, i)
            skips_r.append(r)
            skips_i.append(i)

        # Bottleneck is the last element of skips
        r, i = skips_r.pop(), skips_i.pop()

        # Decoder — consume skips in reverse
        for up in self.ups:
            r, i = up(skips_r.pop(), skips_i.pop(), r, i)

        mask_r, mask_i = self.out_conv(r, i)
        return torch.tanh(mask_r), torch.tanh(mask_i)