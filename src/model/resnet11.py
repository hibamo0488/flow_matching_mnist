import math
import torch
import torch.nn as nn

def timestep_embedding(t, dim=32):
    """Sinusoidal Embedding(いわゆるPosition Embedding)"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half).to(t.device)
    args = t[:, None] * freqs[None, :] * 2 *math.pi
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

# Resnet BasicBlock
class BasicBlock(nn.Module):
    """Resnet構成の基本単位(深くする場合はBottleneckというBlock推奨)"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.norm(x)
        x = self.silu(x)

        x = self.conv2(x)
        x = self.norm(x)
        return self.silu(x + identity)
    
class MnistFM(nn.Module):
    def __init__(self, num_classes: int=10):
        super().__init__()
        self.embed = nn.Embedding(num_classes, 32) # 10×32の行列. 0~9の各埋め込み(32次元)

        # 入力層. stemはraw dataを潜在空間に落とし込むためのテンプレ
        self.stem = nn.Conv2d(1+32+32, 128, 3, padding=1) 

        # resnet中間層
        self.resnet = nn.ModuleList([
            BasicBlock(128) for _ in range(4)
        ])

        # 出力層
        self.out = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def sample_noise(self, shape, device) -> torch.Tensor:
        """標準正規分布のノイズの取得"""
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, batch_size, device):
        """学習時に使用するtime_step(ベータ分布で取得)"""
        time_beta = self._sample_beta(1.5, 1.0, batch_size, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)
    
    def _sample_beta(self, alpha, beta, batch_size, device):
        """ベータ分布"""
        alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
        beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
        dist = torch.distributions.Beta(alpha_t, beta_t)
        return dist.sample((batch_size,))
    
    def forward(self, pseudo_prompt: torch.Tensor, time_step: torch.Tensor, noise: torch.Tensor):
        # pseudo_prompt: (batch,), time_step: (batch,), noise:(batch, 1, 28, 28) 
        B, _, H, W = noise.shape

        # time_step (batch,)をembedding
        # 取得したembedding(batch, 32)を(batch, 32, 28, 28)にブロードキャスト
        t_emb = timestep_embedding(time_step, 32) # (batch, 32)
        t_map = t_emb[:, :, None, None].expand(-1, -1, H, W) # (batch, 32, 28, 28) 

        # 条件(疑似指示文:0~9)をembedding
        # 取得したembedding(batch, 32)を(batch, 32, 28, 28)にブロードキャスト
        cond = self.embed(pseudo_prompt) # (batch, 32)
        cond_map = cond[:, :, None, None].expand(-1, -1, H, W) # (batch, 32, 28, 28)

        # dim=1, すなわちチャネル方向に結合
        input_data = torch.cat([noise, t_map, cond_map], dim=1) # (batch, 1+32+32, 28, 28)

        # 処理
        x = self.stem(input_data)
        for block in self.resnet:
            x = block(x)
        x = self.out(x)
        return x