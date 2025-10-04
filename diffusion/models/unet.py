import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SelfAttention(nn.Module):
    """Self-attention layer."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Compute attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax(q.transpose(-2, -1) @ k * scale, dim=-1)
        
        # Apply attention
        out = v @ attn.transpose(-2, -1)
        out = out.reshape(B, C, H, W)
        
        # Project
        out = self.proj(out)
        
        return x + out

class ResidualBlock(nn.Module):
    """Residual block with time embeddings."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        # Normalization
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Activation and dropout
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        # Add time embedding
        time_emb = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_emb
        
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Skip connection
        return h + self.skip_conv(x)

class UNet(nn.Module):
    """
    Enhanced UNet architecture with time embeddings and attention.
    
    Based on: "Denoising Diffusion Probabilistic Models" by Ho et al.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 model_channels: int = 128, num_res_blocks: int = 2,
                 attention_resolutions: Tuple[int, ...] = (16, 8),
                 dropout: float = 0.1, channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
                 num_heads: int = 8, time_emb_dim: int = 128):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.time_emb_dim = time_emb_dim
        
        # Time embeddings
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = model_channels
        ds = 1
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            # Residual blocks
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                
                # Add attention if resolution matches
                if ds in attention_resolutions:
                    self.down_blocks.append(SelfAttention(ch, num_heads))
            
            # Downsampling (except for last level)
            if i < len(channel_mult) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                ds *= 2
        
        # Middle
        self.middle_block1 = ResidualBlock(ch, ch, time_emb_dim, dropout)
        self.middle_attn = SelfAttention(ch, num_heads)
        self.middle_block2 = ResidualBlock(ch, ch, time_emb_dim, dropout)
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            
            # Residual blocks
            for j in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(ch + out_ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                
                # Add attention if resolution matches
                if ds in attention_resolutions:
                    self.up_blocks.append(SelfAttention(ch, num_heads))
            
            # Upsampling (except for last level)
            if i < len(channel_mult) - 1:
                self.up_samples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                ds //= 2
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # Time embeddings
        time_emb = self.time_emb(time)
        time_emb = self.time_proj(time_emb)
        
        # Input projection
        h = self.input_proj(x)
        
        # Downsampling
        hs = [h]
        for i, block in enumerate(self.down_blocks):
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb)
            else:  # SelfAttention
                h = block(h)
            hs.append(h)
        
        # Downsampling
        for i, sample in enumerate(self.down_samples):
            h = sample(h)
            hs.append(h)
        
        # Middle
        h = self.middle_block1(h, time_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, time_emb)
        
        # Upsampling
        for i, block in enumerate(self.up_blocks):
            if isinstance(block, ResidualBlock):
                # Concatenate with skip connection
                skip_h = hs.pop()
                h = torch.cat([h, skip_h], dim=1)
                h = block(h, time_emb)
            else:  # SelfAttention
                h = block(h)
        
        # Output projection
        h = self.output_proj(h)
        
        return h

class SimpleUNet(nn.Module):
    """
    Simplified UNet for basic experiments.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 hidden_channels: int = 64):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor, time: Optional[torch.Tensor] = None) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(b)
        d1 = self.dec1(d2)
        return d1
