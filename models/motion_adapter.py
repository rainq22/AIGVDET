import torch
import torch.nn as nn


class MotionAdapter(nn.Module):
    def __init__(self, llm_hidden_size: int, num_tokens: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 8, 8)),
        )
        self.proj = nn.Linear(64, llm_hidden_size)

    def forward(self, flow_residuals: torch.Tensor) -> torch.Tensor:
        # flow_residuals: [B, T-2, 2, H, W]
        x = flow_residuals.permute(0, 2, 1, 3, 4)
        features = self.encoder(x)
        bsz = features.shape[0]
        features = features.view(bsz, 64, -1).permute(0, 2, 1)
        motion_embeds = self.proj(features)
        return motion_embeds
