from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    """Single-group normalization used by the restoration transformer blocks."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        head_dim = channels // self.heads
        q = q.reshape(batch, self.heads, head_dim, height * width)
        k = k.reshape(batch, self.heads, head_dim, height * width)
        v = v.reshape(batch, self.heads, head_dim, height * width)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        output = (attention.softmax(dim=-1) @ v).reshape(
            batch, channels, height, width
        )
        return self.project(output)


class GatedFFN(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.in_project = nn.Conv2d(channels, hidden * 2, 1, bias=False)
        self.depthwise = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            3,
            padding=1,
            groups=hidden * 2,
            bias=False,
        )
        self.out_project = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = self.depthwise(self.in_project(x)).chunk(2, dim=1)
        return self.out_project(F.gelu(first) * second)


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attention = ChannelAttention(channels, heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GatedFFN(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


def make_blocks(channels: int, count: int, heads: int) -> nn.Sequential:
    return nn.Sequential(*(TransformerBlock(channels, heads) for _ in range(count)))


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class AnchorIR(nn.Module):
    """Inference restoration backbone described in the AnchorIR paper.

    Input is an RGB tensor shaped (B, 3, H, W). Height and width must be
    divisible by four. The output is the residual-added restored image.
    """

    def __init__(self, dim: int = 48) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, dim, 3, padding=1, bias=False)
        self.enc1 = make_blocks(dim, 2, 1)
        self.down1 = Downsample(dim)
        self.enc2 = make_blocks(dim * 2, 2, 2)
        self.down2 = Downsample(dim * 2)
        self.latent = make_blocks(dim * 4, 3, 4)
        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.dec2 = make_blocks(dim * 2, 2, 2)
        self.up1 = Upsample(dim * 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.dec1 = make_blocks(dim, 2, 1)
        self.output = nn.Conv2d(dim, 3, 3, padding=1)

    def forward_with_latent(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        level1 = self.enc1(self.patch(image))
        level2 = self.enc2(self.down1(level1))
        latent = self.latent(self.down2(level2))

        decoded2 = self.up2(latent)
        decoded2 = self.dec2(self.reduce2(torch.cat((decoded2, level2), dim=1)))
        decoded1 = self.up1(decoded2)
        decoded1 = self.dec1(self.reduce1(torch.cat((decoded1, level1), dim=1)))
        restored = image + self.output(decoded1)
        return restored, latent

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        restored, _ = self.forward_with_latent(image)
        return restored


class DegradationClassifier(nn.Module):
    """Training-only 7-way classifier attached to the latent features."""

    def __init__(self, channels: int = 192, classes: int = 7) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, 96),
            nn.GELU(),
            nn.Linear(96, classes),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        pooled = latent.mean(dim=(-2, -1))
        return self.mlp(pooled)


class AnchorIRTrainingModel(nn.Module):
    """Restoration backbone plus the paper's training-only degradation head."""

    def __init__(self, dim: int = 48, degradation_count: int = 7) -> None:
        super().__init__()
        self.restorer = AnchorIR(dim=dim)
        self.degradation_head = DegradationClassifier(
            channels=dim * 4, classes=degradation_count
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        restored, latent = self.restorer.forward_with_latent(image)
        return {
            "restored": restored,
            "degradation_logits": self.degradation_head(latent),
        }


class SourceVerifier(nn.Module):
    """Pretrained source verifier used as a separate frozen training module."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 192),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.projector = nn.Sequential(
            nn.Linear(192, 192),
            nn.GELU(),
            nn.Linear(192, embedding_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        embedding = self.projector(self.encoder(image))
        return F.normalize(embedding, dim=-1)


class SourceVerifierPretrainingModel(nn.Module):
    """Source-classification head used only when pretraining the verifier."""

    def __init__(self, source_count: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.verifier = SourceVerifier(embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, source_count)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.verifier(image)
        return {"embedding": embedding, "source_logits": self.classifier(embedding)}


def freeze_verifier(verifier: SourceVerifier) -> SourceVerifier:
    verifier.eval()
    for parameter in verifier.parameters():
        parameter.requires_grad_(False)
    return verifier
