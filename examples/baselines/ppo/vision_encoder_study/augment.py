"""数据增强 (Data Augmentation) for 视觉强化学习.

实现 DrQ (Kostrikov et al. 2021) 风格的图像增强，用于提升视觉 PPO 的样本效率。
增强在 GPU 上对一个 batch 的图像 (B, H, W, C) uint8 直接操作，开销很小。

用法（在 ppo_rgb_encoders.py 的 minibatch 更新前对图像 obs 应用）：

    aug = RandomShiftAug(pad=4)
    obs["rgb"] = aug(obs["rgb"])          # 仅增强图像，state 不动

两种增强：
- RandomShiftAug：随机平移裁剪（DrQ 的核心增强，对操作任务最有效、最稳）。
- ColorJitterAug：亮度/对比度抖动（可选，增加视觉鲁棒性）。
- AugPipeline：把多个增强串起来，由训练脚本的 --use_aug 开关控制。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomShiftAug(nn.Module):
    """DrQ 风格随机平移裁剪。

    先在四周做 replicate padding，再随机裁剪回原尺寸，等价于随机平移图像。
    对 batch 中每张图采样独立的平移量。

    输入/输出: (B, H, W, C) uint8。
    """

    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        assert x.dtype == torch.uint8, "期望 uint8 图像输入"
        b, h, w, c = x.shape
        # 转 (B,C,H,W) float 做 padding/采样
        img = x.permute(0, 3, 1, 2).float()
        img = F.pad(img, (self.pad,) * 4, mode="replicate")

        # 为每张图采样独立的裁剪起点
        eps_h = torch.randint(0, 2 * self.pad + 1, (b,), device=x.device)
        eps_w = torch.randint(0, 2 * self.pad + 1, (b,), device=x.device)

        # 用 grid_sample 实现可向量化的逐样本平移裁剪
        padded_h, padded_w = h + 2 * self.pad, w + 2 * self.pad
        # 构造每张图的基础网格 (归一化到 [-1,1])
        ys = torch.arange(h, device=x.device).float()
        xs = torch.arange(w, device=x.device).float()
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (h,w)
        grid_y = grid_y.unsqueeze(0) + eps_h.view(b, 1, 1)       # (b,h,w)
        grid_x = grid_x.unsqueeze(0) + eps_w.view(b, 1, 1)
        # 归一化到 [-1,1]
        grid_y = 2.0 * grid_y / (padded_h - 1) - 1.0
        grid_x = 2.0 * grid_x / (padded_w - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)             # (b,h,w,2)

        out = F.grid_sample(img, grid, mode="nearest", align_corners=True)
        # 转回 (B,H,W,C) uint8
        return out.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)


class ColorJitterAug(nn.Module):
    """简单的亮度/对比度抖动，对 batch 逐样本采样系数。

    输入/输出: (B, H, W, C) uint8。
    """

    def __init__(self, brightness=0.2, contrast=0.2):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, x):
        assert x.dtype == torch.uint8
        b = x.shape[0]
        img = x.float()
        # 亮度：乘性系数 in [1-b, 1+b]
        bf = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * self.brightness
        img = img * bf
        # 对比度：围绕每张图均值缩放
        cf = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * self.contrast
        mean = img.mean(dim=(1, 2, 3), keepdim=True)
        img = (img - mean) * cf + mean
        return img.round().clamp(0, 255).to(torch.uint8)


class AugPipeline(nn.Module):
    """串联多个增强。空 pipeline 等价于恒等映射。"""

    def __init__(self, augs):
        super().__init__()
        self.augs = nn.ModuleList(augs)

    def forward(self, x):
        for aug in self.augs:
            x = aug(x)
        return x


def build_aug(use_aug, pad=4, color_jitter=False):
    """根据开关构造增强 pipeline。use_aug=False 返回 None (训练脚本里跳过)。"""
    if not use_aug:
        return None
    augs = [RandomShiftAug(pad=pad)]
    if color_jitter:
        augs.append(ColorJitterAug())
    return AugPipeline(augs)
