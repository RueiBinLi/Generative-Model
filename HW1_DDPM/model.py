import torch.nn as nn
import tqdm
from model_utils import *

'''
| Layer                | Input Size (Channels x H x W)         | Output Size (Channels x H x W)        | Notes                                                                  |
|----------------------|---------------------------------------|---------------------------------------|------------------------------------------------------------------------|
| `self.inc(x)`        | `c_in x H x W`                        | `32 x H x W`                          |
| `self.down1(x1)`     | `32 x H x W`                          | `64 x H/2 x W/2`                      |
| `self.sa1(x2)`       | `64 x H/2 x W/2`                      | `64 x H/2 x W/2`                      |
| `self.down2(x2)`     | `64 x H/2 x W/2`                      | `128 x H/4 x W/4`                     |
| `self.sa2(x3)`       | `128 x H/4 x W/4`                     | `128 x H/4 x W/4`                     |
| `self.down3(x3)`     | `128 x H/4 x W/4`                     | `128 x H/8 x W/8`                     |
| `self.sa3(x4)        | `128 x H/8 x W/8`                     | `128 x H/8 x W/8`                     |
| `self.bot1(x4)       | `128 x H/8 x W/8`                     | `256 x H/8 x W/8`                     |
| `self.bot2(x4)       | `256 x H/8 x W/8`                     | `256 x H/8 x W/8`                     |
| `self.bot3(x4)       | `256 x H/8 x W/8`                     | `128 x H/8 x W/8`                     |
| `self.up1(x4, x3)    | `128 x H/8 x W/8` & `128 x H/4 x W/4` | `128 x H/4 x W/4`                     |
| `self.sa4(x)         | `128 x H/4 x W/4`                     | `128 x H/4 x W/4`                     |
| `self.up2(x3, x2)    | `128 x H/4 x W/4` & `64 x H/2 x W/2`  | `64 x H/2 x H/2`                      |
| `self.sa5(x)         | `64 x H/2 x W/2`                      | `64 x H/2 x W/2`                      |
| `self.up3(x2, x1)    | `64 x H/2 x W/2`  & `32 x H x W`      | `32 x H x W`                          |
| `self.sa6(x)         | `32 x H x W`                          | `32 x H x W`                          |
| `self.outc(x)        | `32 x H x W`                          | `c_out x H x W`                       |
'''

class UNet(nn.Module):
    # num_classes tell the model how may different categories of images it should learn to generate
    def __init__(self, c_in=3, c_out=3, img_size=64, time_dim=256, device='cuda'):
        super().__init__()
        self.img_size = img_size
        self.time_dim = time_dim
        self.device = device

        self.inc = DoubleConv(in_channels=3, out_channels=64) # (3, 64, 64) -> (64, 64, 64)
        self.down1 = Down(in_channels=64, out_channels=128) # (64, 64, 64) -> (128, 32, 32)
        self.sa1 = SelfAttention(channels=128, size=int(img_size/2)) # (128, 32, 32) -> (128, 32, 32)
        self.down2 = Down(in_channels=128, out_channels=256) # (128, 32, 32) -> (256, 16, 16)
        self.sa2 = SelfAttention(channels=256, size=int(img_size/4)) # (256, 16, 16) -> (256, 16, 16)
        self.down3 = Down(in_channels=256, out_channels=256) # (256, 16, 16) -> (256, 8, 8)
        self.sa3 = SelfAttention(channels=256, size=int(img_size/8)) # (256, 8, 8) -> (256, 8, 8)

        self.bot1 = DoubleConv(in_channels=256, out_channels=512) # (256, 8, 8) -> (512, 8, 8)
        self.bot2 = DoubleConv(in_channels=512, out_channels=512) # (512, 8, 8) -> (512, 8, 8)
        self.bot3 = DoubleConv(in_channels=512, out_channels=256) # (512, 8, 8) -> (256, 8, 8)

        self.up1 = Up(in_channels=512, out_channels=128) # (256, 8, 8) -> (256, 16, 16)
        self.sa4 = SelfAttention(channels=128, size=int(img_size/4)) # (128, 16, 16) -> (128, 16, 16)
        self.up2 = Up(in_channels=256, out_channels=64) # (256, 16, 16) -> (128, 32, 32)
        self.sa5 = SelfAttention(channels=64, size=int(img_size/2)) # (64, 32, 32) -> (64, 32, 32)
        self.up3 = Up(in_channels=128, out_channels=64) # (128, 32, 32) -> (64, 64, 64)
        self.sa6 = SelfAttention(channels=64, size=img_size) # (64, 64, 64) -> (64, 64, 64)
        self.outc = nn.Conv2d(in_channels=64, out_channels=c_out, kernel_size=1) # (64, 64, 64) -> (3, 64, 64)
    
    def pos_encoding(self, t, channels):
        # sinusoidal embedding formula
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)) # 1.0 / (10000 ** [0/128, 2/128, ..., 126/128])
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq) # even dimension
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq) # odd dimension
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        t = t.unsqueeze(-1).to(torch.float) # [t] -> [t,]
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)

        return output
    
class Diffusion:
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=2e-2, img_size=32, device='cuda'):
        self.steps = steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.beta = self.linear_schedular().to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    # the hyperparamter which decides the parameter to noise images
    def linear_schedular(self):
        return torch.linspace(self.beta_start, self.beta_end, self.steps) # to get beta

    def noise_images(self, x, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None] # [batch_size, 1, 1, 1]. The batch_size is from datasets
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None] # [batch_size, 1, 1, 1]
        z = torch.randn_like(x) # find a random number from a standard normal distribution. The format of the z is like x. x:[batch, channels, height, width]
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * z, z # z is the noise we add to image. We need it to calculate loss
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.steps, size=(n,)) # return the random timesteps for training
    
    def sample(self, model, n, image_channels):
        result = []
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, image_channels, self.img_size, self.img_size)).to(self.device) # get a random tensor from standard normal distribution numbers
            for i in tqdm(reversed(range(1, self.noise_images)), position=0):
                t = (torch.ones() * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_bar = self.alpha_bar[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) + torch.sqrt(beta) * noise
                if (i % 100 == 0):
                    result.append(x)
        model.train()
        result.append(x)
        result = torch.cat(result)
        result = (result.clamp(-1,1) + 1) / 2 # denormalize the value. Clip the pixel value to [-1,1]. (+1) Shift the value to [0,2]. (/2) Scales the range down to [0,1]
        result = (result * 255).type(torch.uint8) # converts the pixel value to a standard 8-bit image format 
        return result