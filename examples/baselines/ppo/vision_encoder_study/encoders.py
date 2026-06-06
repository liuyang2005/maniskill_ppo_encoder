"""视觉编码器库 (Vision Encoder Zoo).

本文件为保研作品集项目「面向机器人操作的视觉编码器对比研究」的核心组件。
所有编码器共享统一接口，可直接替换官方 ppo_rgb.py 中写死的 NatureCNN：

    encoder = ENCODER_REGISTRY[name](sample_obs, **kwargs)
    feat = encoder(observations)          # observations: dict, 含 "rgb" (B,H,W,C uint8) 及可选 "state"
    assert feat.shape[1] == encoder.out_features

设计约束：
- 输入图像为 (B, H, W, C) 的 uint8 (ManiSkill rgb obs)，编码器内部统一转 (B,C,H,W) 并归一化到 [0,1]。
- 图像分支输出 256 维；若 obs 含 "state"，则额外用一个 Linear(state_dim, 256) 分支，拼接后 out_features = 512。
  （与官方实现保持一致，保证各编码器之间对比公平。）
- 为适配 6GB 显存，各编码器参数量都刻意控制在较小规模。
"""
import math

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化，与官方 ppo_rgb.py 一致。"""
    if hasattr(layer, "weight") and layer.weight is not None and layer.weight.dim() >= 2:
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


# 各编码器图像分支统一的输出维度
FEATURE_SIZE = 256


class _StateMixin:
    """为编码器提供可选的 state 分支，避免在每个编码器里重复代码。"""

    def _build_state_branch(self, sample_obs):
        self.has_state = "state" in sample_obs
        if self.has_state:
            state_size = sample_obs["state"].shape[-1]
            self.state_extractor = nn.Linear(state_size, FEATURE_SIZE)
            return FEATURE_SIZE
        return 0

    def _encode_state(self, observations):
        if self.has_state:
            return self.state_extractor(observations["state"])
        return None

    @staticmethod
    def _prep_image(observations):
        """(B,H,W,C) uint8 -> (B,C,H,W) float in [0,1]."""
        rgb = observations["rgb"]
        return rgb.float().permute(0, 3, 1, 2) / 255.0


def _infer_flatten_dim(cnn, sample_obs):
    """用一个测试张量推断 CNN flatten 后的维度（与官方做法一致）。"""
    with torch.no_grad():
        img = sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu() / 255.0
        return cnn(img).shape[1]


# --------------------------------------------------------------------------- #
# 1) NatureCNN —— 官方基线 (Mnih et al. 2015 的 DQN backbone)
# --------------------------------------------------------------------------- #
class NatureCNN(nn.Module, _StateMixin):
    """官方 ppo_rgb.py 使用的编码器，作为对比基线。"""

    def __init__(self, sample_obs, **kwargs):
        super().__init__()
        in_channels = sample_obs["rgb"].shape[-1]
        cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        n_flatten = _infer_flatten_dim(cnn, sample_obs)
        self.image_extractor = nn.Sequential(
            cnn, nn.Linear(n_flatten, FEATURE_SIZE), nn.ReLU()
        )
        self.out_features = FEATURE_SIZE + self._build_state_branch(sample_obs)

    def forward(self, observations):
        feats = [self.image_extractor(self._prep_image(observations))]
        state = self._encode_state(observations)
        if state is not None:
            feats.append(state)
        return torch.cat(feats, dim=1)


# --------------------------------------------------------------------------- #
# 2) ImpalaCNN —— 带残差块的 CNN (Espeholt et al. 2018, DrQ/Procgen 常用)
#    通常比 NatureCNN 样本效率更好，参数量适中。
# --------------------------------------------------------------------------- #
class _ImpalaResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = self.conv0(torch.relu(x))
        out = self.conv1(torch.relu(out))
        return x + out


class _ImpalaConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res0 = _ImpalaResidualBlock(out_channels)
        self.res1 = _ImpalaResidualBlock(out_channels)

    def forward(self, x):
        x = self.pool(self.conv(x))
        x = self.res0(x)
        x = self.res1(x)
        return x


class ImpalaCNN(nn.Module, _StateMixin):
    """IMPALA 风格的残差 CNN。channels 默认 (16, 32, 32) 以控制显存。"""

    def __init__(self, sample_obs, channels=(16, 32, 32), **kwargs):
        super().__init__()
        in_channels = sample_obs["rgb"].shape[-1]
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers.append(_ImpalaConvSequence(c_in, c_out))
            c_in = c_out
        layers += [nn.ReLU(), nn.Flatten()]
        cnn = nn.Sequential(*layers)
        n_flatten = _infer_flatten_dim(cnn, sample_obs)
        self.image_extractor = nn.Sequential(
            cnn, nn.Linear(n_flatten, FEATURE_SIZE), nn.ReLU()
        )
        self.out_features = FEATURE_SIZE + self._build_state_branch(sample_obs)

    def forward(self, observations):
        feats = [self.image_extractor(self._prep_image(observations))]
        state = self._encode_state(observations)
        if state is not None:
            feats.append(state)
        return torch.cat(feats, dim=1)


# --------------------------------------------------------------------------- #
# 3) SmallViT —— 轻量 Vision Transformer (patch embedding + 几层 encoder)
#    参数量受控以适配；研究 attention-based 表征是否优于卷积。
# --------------------------------------------------------------------------- #
class SmallViT(nn.Module, _StateMixin):
    def __init__(
        self,
        sample_obs,
        patch_size=8,
        dim=128,
        depth=3,
        heads=4,
        mlp_ratio=2.0,
        **kwargs,
    ):
        super().__init__()
        in_channels = sample_obs["rgb"].shape[-1]
        h, w = sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2]
        assert h % patch_size == 0 and w % patch_size == 0, (
            f"图像尺寸 {h}x{w} 必须能被 patch_size={patch_size} 整除"
        )
        n_patches = (h // patch_size) * (w // patch_size)

        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.image_proj = nn.Sequential(nn.Linear(dim, FEATURE_SIZE), nn.ReLU())
        self.out_features = FEATURE_SIZE + self._build_state_branch(sample_obs)

    def forward(self, observations):
        x = self._prep_image(observations)           # (B,C,H,W)
        x = self.patch_embed(x)                       # (B,dim,h',w')
        x = x.flatten(2).transpose(1, 2)              # (B,N,dim)
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])                        # 取 cls token
        feats = [self.image_proj(x)]
        state = self._encode_state(observations)
        if state is not None:
            feats.append(state)
        return torch.cat(feats, dim=1)


# --------------------------------------------------------------------------- #
# 3b) ConvStemViT —— 针对 ViT 失效问题提出的改进编码器
#
# 动机：SmallViT 在视觉操作 RL 中完全学不会 (success=0)。原因是 ViT 缺乏 CNN 的
#   归纳偏置（局部性、平移不变性），且一次性大 patch embedding 把图像粗暴切块、
#   丢失了操作任务关键的物体边缘/接触等细节，在小数据 RL 下难以从零学到有用表征。
#
# 改进：参考 Xiao et al. 2021 "Early Convolutions Help Transformers See Better"，
#   用一个轻量卷积 stem（几层 3x3 conv + stride 下采样）替代大 patch embedding。
#   卷积 stem 提供局部归纳偏置、提取低层视觉特征，再交给 transformer 做全局推理。
#
# 设计要点：与 SmallViT 仅在"图像 token 化方式"上不同（patch_embed -> conv_stem），
#   transformer / cls / pos_embed / 输出头完全一致 —— 这样消融时能干净地归因到 stem。
# --------------------------------------------------------------------------- #
class _ConvStem(nn.Module):
    """轻量卷积 stem：把 (B,3,H,W) 下采样并升维到 (B,dim,H/r,W/r)。

    用 3x3 卷积逐步下采样（每层 stride=2），通道翻倍，最后 1x1 conv 投到 dim。
    例如 64x64 输入、3 层 stride2 -> 8x8 的特征图（等效 patch_size=8，但带归纳偏置）。
    """

    def __init__(self, in_channels, dim, n_downsample=3):
        super().__init__()
        layers = []
        c = 32
        c_in = in_channels
        for i in range(n_downsample):
            layers += [
                nn.Conv2d(c_in, c, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, c) if c >= 8 else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            c_in = c
            c = min(c * 2, dim)
        # 投影到 transformer 的 token 维度
        layers.append(nn.Conv2d(c_in, dim, kernel_size=1))
        self.stem = nn.Sequential(*layers)

    def forward(self, x):
        return self.stem(x)  # (B, dim, H/2^n, W/2^n)


class ConvStemViT(nn.Module, _StateMixin):
    """卷积 stem + Transformer。本项目用于"救活" ViT 的核心改进编码器。"""

    def __init__(
        self,
        sample_obs,
        dim=128,
        depth=3,
        heads=4,
        mlp_ratio=2.0,
        n_downsample=3,
        **kwargs,
    ):
        super().__init__()
        in_channels = sample_obs["rgb"].shape[-1]
        h, w = sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2]

        # 卷积 stem 替代 patch embedding
        self.conv_stem = _ConvStem(in_channels, dim, n_downsample=n_downsample)
        # 推断 stem 输出的 token 数
        with torch.no_grad():
            dummy = sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()[:1] / 255.0
            feat = self.conv_stem(dummy)
            n_patches = feat.shape[2] * feat.shape[3]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.image_proj = nn.Sequential(nn.Linear(dim, FEATURE_SIZE), nn.ReLU())
        self.out_features = FEATURE_SIZE + self._build_state_branch(sample_obs)

    def forward(self, observations):
        x = self._prep_image(observations)        # (B,C,H,W)
        x = self.conv_stem(x)                      # (B,dim,h',w') —— 带归纳偏置的 token 化
        x = x.flatten(2).transpose(1, 2)           # (B,N,dim)
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])                     # cls token
        feats = [self.image_proj(x)]
        state = self._encode_state(observations)
        if state is not None:
            feats.append(state)
        return torch.cat(feats, dim=1)


# --------------------------------------------------------------------------- #
# 4) ResNet18 —— torchvision 预训练 backbone，可冻结
#    对比「ImageNet 预训练表征」与「从头训练」在操作任务上的差异。
# --------------------------------------------------------------------------- #
class ResNet18Encoder(nn.Module, _StateMixin):
    def __init__(self, sample_obs, pretrained=True, freeze=False, **kwargs):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        # 去掉最后的全连接层，保留到 avgpool (输出 512 维)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        backbone_out = 512

        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # ImageNet 归一化常数
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.image_proj = nn.Sequential(nn.Linear(backbone_out, FEATURE_SIZE), nn.ReLU())
        self.out_features = FEATURE_SIZE + self._build_state_branch(sample_obs)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()   # 冻结时 backbone 始终 eval (保持 BN 统计量稳定)
        return self

    def forward(self, observations):
        x = self._prep_image(observations)
        x = (x - self.mean) / self.std
        if self.frozen:
            with torch.no_grad():
                feat = self.backbone(x).flatten(1)
        else:
            feat = self.backbone(x).flatten(1)
        feats = [self.image_proj(feat)]
        state = self._encode_state(observations)
        if state is not None:
            feats.append(state)
        return torch.cat(feats, dim=1)


# --------------------------------------------------------------------------- #
# 注册表：训练脚本通过 --encoder 名字索引
# --------------------------------------------------------------------------- #
ENCODER_REGISTRY = {
    "nature": NatureCNN,
    "impala": ImpalaCNN,
    "smallvit": SmallViT,
    # [STUDY] ViT 消融变体：更小 patch（8->4），研究 patch 粒度对 ViT 的影响
    "smallvit_p4": lambda sample_obs, **kw: SmallViT(sample_obs, patch_size=4, **kw),
    # [STUDY] 核心改进：卷积 stem + Transformer，用于"救活" ViT
    "convstem_vit": ConvStemViT,
    "resnet18": lambda sample_obs, **kw: ResNet18Encoder(
        sample_obs, pretrained=True, **kw
    ),
    "resnet18_scratch": lambda sample_obs, **kw: ResNet18Encoder(
        sample_obs, pretrained=False, **kw
    ),
}


def build_encoder(name, sample_obs, **kwargs):
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"未知编码器 '{name}'，可选: {list(ENCODER_REGISTRY.keys())}"
        )
    return ENCODER_REGISTRY[name](sample_obs, **kwargs)


def count_parameters(module):
    """返回 (可训练参数量, 总参数量)。"""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total
