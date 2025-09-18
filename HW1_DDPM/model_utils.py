import torch
import torch.nn as nn
import torch.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels), # (GroupNumbers, Channels)
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        output = self.double_conv(x)
        return output
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256): # emb_dim is hyperparameter
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            # DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        # convert the dimension of emb_dim to out_channels
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim, 
                out_channels,
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)

        # get the time embedding
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # shape[-2] counts the array from back (batch_size, out_channels, H, W)

        # add the time embedding to the original image to tell the model how strong the noise is
        return x + emb
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256): # emb_dim is hyperparamter
        super().__init__()

        # scale_factor=2, 16x16 -> 32x32 which double the height and width
        # mode="bilinear", the new pixel is calculated as a weighted average of the 4 nearest original pixels
        # When align_corner set to True, it treats the corner pixels of the input and output as perfectly aligned
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1) # skip connection, skip_x is from encoder. dim = 1 represent the skip_x concat x with dimension 1 channels. [batch_size, channels, H, W]
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size

        # num_heads=4 means there is 4 specialists
        # batch_first=True tells the program that the input is (batch, sequence, features)
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.reshape(-1, self.channels, self.size * self.size).swapaxes(1, 2) # Reshape x from (batch, channels, H, W) to (batch, H*W, channels)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln) # Three input: Query, Key and Value. To find internal relationship
        attention_value = attention_value + x # residual connection
        attention_value = self.ff_self(attention_value) + attention_value # Feed-Forward Network and another residual connection
        return attention_value.swapaxes(2, 1).reshape(-1, self.channels, self.size, self.size) # Reshape to (batch, channels, H, W)