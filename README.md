# 基于ManiSkill的视觉编码器对机器人操作强化学习的影响研究

> A Study of Vision Encoders for Robotic Manipulation RL under Limited Compute
>
> 基于 [ManiSkill](https://github.com/haosulab/ManiSkill) 仿真平台与 PPO 算法
>
> 本项目基于官方 ManiSkill 仓库 fork 开发，代码位于 `examples/baselines/ppo/vision_encoder_study/`

在单张消费级 GPU 的算力约束下，系统对比不同视觉编码器（CNN / ViT / 预训练 ResNet）在机器人操作任务上的样本效率、成功率与训练开销；进一步诊断轻量 Vision Transformer 的失效成因，提出改进并通过消融验证。

本项目改编自官方 `examples/baselines/ppo/ppo_rgb.py`，所有改动均带 `# [STUDY]` 标注，自包含于本目录，不修改 ManiSkill 源码。

---

## 1. 研究问题

1. 相同算法与超参下，不同视觉编码器在机器人操作任务上如何权衡样本效率、成功率与训练吞吐？
2. 当轻量 ViT 在该设定下失效时，根本原因是什么？能否针对性地改进？

## 2. 方法

### 2.1 公平对比框架
所有编码器共享统一接口：相同图像观测，图像分支统一输出 256 维、与状态分支 256 维拼接为 512 维，下游 actor/critic 完全一致。编码器是唯一受控变量。

### 2.2 对比的编码器（`encoders.py`）
| 编码器 | 说明 | 可训练参数 |
|--------|------|-----------|
| `nature` | Nature-DQN 卷积骨干（基线）| ~0.15M |
| `impala` | IMPALA 残差卷积网络 | ~0.40M |
| `smallvit` | 轻量 Vision Transformer | ~0.47M |
| `convstem_vit` | **本项目改进**：卷积词元化 + Transformer | ~0.55M |
| `resnet18` | ImageNet 预训练骨干（可冻结）| ~0.14M（可训练）|

### 2.3 核心改进（针对 ViT 失效）
- **ConvStem**：以卷积层替代大 patch embedding，注入卷积的归纳偏置（依据 Xiao et al. 2021）。
- **LR warmup**：训练初期学习率线性升温，避免早期大梯度破坏自注意力结构。

### 2.4 数据增强（`augment.py`）
DrQ 风格随机平移裁剪，GPU 上向量化处理。

## 3. 实验设置
- 硬件：单张 NVIDIA RTX 4050 Laptop（6GB）
- 任务：PushCube-v1（推方块）
- 观测：48×48 RGB + 机器人本体状态
- 算法：PPO，所有编码器共用同一组超参
- 训练：每配置 1.5M 环境步（单 seed）

## 4. 主要结果

### 4.1 编码器横向对比（`figures/sample_efficiency.png`, `final_success.png`）
| 编码器 | 最终成功率 | 训练吞吐(SPS) | 备注 |
|--------|-----------|--------------|------|
| nature | **1.00** | **2194** | 最快最稳，性价比最优 |
| impala | 1.00 | 962 | 样本效率高，末态波动 |
| resnet18(冻结) | 0.50(峰值1.0) | 1299 | 预训练特征不完全适配 |
| smallvit | **0.00** | 1036 | 完全失效 |

> 关键观察：训练速度由**计算量(FLOPs)**而非参数量主导——impala 参数仅为 nature 的 2.7 倍，但单次前向慢约一个数量级。

### 4.2 数据增强（`figures/aug_comparison.png`）
对 impala 施加随机平移增强后，回合末成功率由 0.25 提升至 0.62，主要改善策略稳定性。

### 4.3 救活 ViT 的消融（`figures/vit_ablation.png`，核心结果）
| 配置 | ConvStem | warmup | 成功率(峰值) |
|------|:---:|:---:|:---:|
| smallvit 基线 | ✗ | ✗ | 0.00 |
| 仅 ConvStem | ✓ | ✗ | 0.00 |
| 仅 warmup | ✗ | ✓ | 0.00 |
| **ConvStem + warmup** | ✓ | ✓ | **0.62** |

**结论：两项改进缺一不可，唯有协同方能将 ViT 从完全失效（0）救至可用（0.62）。** 救活后的 ViT 仍不及同等规模 CNN 稳定，说明卷积归纳偏置的优势在小算力操作 RL 下难以被轻量 ViT 替代。

## 5. 复现

```bash
pip install -r requirements.txt

cd examples/baselines/ppo/vision_encoder_study

# 单次训练（示例：改进的 ConvStemViT + warmup）
python ppo_rgb_encoders.py --env_id=PushCube-v1 --encoder=convstem_vit \
    --warmup_iters=15 --cam_size=48 --num_envs=128 --total_timesteps=1500000

# 实验矩阵
bash run_experiments.sh main        # 编码器横向对比
bash run_experiments.sh aug         # 数据增强对比
bash run_vit_ablation.sh            # ViT 消融

# 生成对比图表
python analyze.py --env PushCube-v1
```

## 6. 文件结构
```
vision_encoder_study/
├── encoders.py            # 编码器库（含改进的 ConvStemViT）
├── augment.py             # DrQ 数据增强
├── ppo_rgb_encoders.py    # 训练入口（改编自官方 ppo_rgb.py）
├── run_experiments.sh     # 主实验脚本（6GB 配置）
├── run_experiments_4090.sh# 高配实验脚本（大显存）
├── run_vit_ablation.sh    # ViT 消融脚本
├── analyze.py             # 日志解析 + 出图
├── figures/               # 结果图表
└── runs/                  # 训练日志（TensorBoard）
```

## 7. 参考
- ManiSkill：GPU 并行机器人仿真平台
- Schulman et al. 2017, PPO
- Dosovitskiy et al. 2021, Vision Transformer
- Xiao et al. 2021, *Early Convolutions Help Transformers See Better*
- Kostrikov et al. 2021, DrQ


# ManiSkill 3


![teaser](figures/teaser.jpg)
<p style="text-align: center; font-size: 0.8rem; color: #999;margin-top: -1rem;">Sample of environments/robots rendered with ray-tracing. Scene datasets sourced from AI2THOR and ReplicaCAD</p>

[![Downloads](https://static.pepy.tech/badge/mani_skill)](https://pepy.tech/project/mani_skill)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haosulab/ManiSkill/blob/main/examples/tutorials/1_quickstart.ipynb)
[![PyPI version](https://badge.fury.io/py/mani-skill.svg)](https://badge.fury.io/py/mani-skill)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://maniskill.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/discord/996566046414753822?logo=discord)](https://discord.gg/x8yUZe5AdN)

ManiSkill is an open-source framework for robot simulation and training powered by [SAPIEN](https://sapien.ucsd.edu/), with a strong focus on manipulation skills. Among its features include:
- GPU parallelized visual data collection system. On the high end you can collect RGBD + Segmentation data at 30,000+ FPS on a 4090 GPU
- GPU parallelized simulation, enabling high throughput state-based synthetic data collection in simulation
- GPU parallelized heterogeneous simulation, where every parallel environment has a completely different scene/set of objects
- Example tasks cover a wide range of different robot embodiments (humanoids, mobile manipulators, single-arm robots) as well as a wide range of different tasks (table-top, drawing/cleaning, dexterous manipulation)
- Flexible and simple task building API that abstracts away much of the complex GPU memory management code via an object oriented design
- Real2sim environments for scalably evaluating real-world policies 100x faster via GPU simulation.
- Sim2real examples for deploying policies trained in simulation to the real world
- Many tuned robot learning baselines in Reinforcement Learning (e.g. PPO, SAC, [TD-MPC2](https://github.com/nicklashansen/tdmpc2)), Imitation Learning (e.g. Behavior Cloning, [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)), and large Vision Language Action (VLA) models (e.g. [Octo](https://github.com/octo-models/octo), [RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer), [RT-x](https://robotics-transformer-x.github.io/))

For more details we encourage you to take a look at our [paper](https://arxiv.org/abs/2410.00425), published at [RSS 2025](https://roboticsconference.org/).

Please refer to our [documentation](https://maniskill.readthedocs.io/en/latest/user_guide) to learn more information from tutorials on building tasks to sim2real to running baselines. If you find any bugs or have any feature requests please post them to our [GitHub issues](https://github.com/haosulab/ManiSkill/issues/) or discuss about them on [GitHub discussions](https://github.com/haosulab/ManiSkill/discussions/). We also have a [Discord Server](https://discord.gg/x8yUZe5AdN) through which we make announcements and discuss about ManiSkill.

Users looking for the original ManiSkill2 can find the commit for that codebase at the [v0.5.3 tag](https://github.com/haosulab/ManiSkill/tree/v0.5.3)

## Installation
Installation of ManiSkill is extremely simple, you only need to run a few pip installs and setup Vulkan for rendering.

```bash
# install the package
pip install --upgrade mani_skill
# install a version of torch that is compatible with your system
pip install torch
```

Finally you also need to set up Vulkan with [instructions here](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)

For more details about installation (e.g. from source, or doing troubleshooting) see [the documentation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html
)

## Getting Started

To get started, check out the quick start documentation: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html

We also have a quick start [colab notebook](https://colab.research.google.com/github/haosulab/ManiSkill/blob/main/examples/tutorials/1_quickstart.ipynb) that lets you try out GPU parallelized simulation without needing your own hardware. Everything is runnable on Colab free tier.

For a full list of example scripts you can run, see [the docs](https://maniskill.readthedocs.io/en/latest/user_guide/demos/index.html).

## System Support

We currently best support Linux based systems. There is limited support for windows and MacOS at the moment. We are working on trying to support more features on other systems but this may take some time. Most constraints stem from what the [SAPIEN](https://github.com/haosulab/SAPIEN/) package is capable of supporting.

| System / GPU         | CPU Sim | GPU Sim | Rendering |
| -------------------- | ------- | ------- | --------- |
| Linux / NVIDIA GPU   | ✅      | ✅      | ✅        |
| Windows / NVIDIA GPU | ✅      | ❌      | ✅        |
| Windows / AMD GPU    | ✅      | ❌      | ✅        |
| WSL / Anything       | ✅      | ❌      | ❌        |
| MacOS / Anything     | ✅      | ❌      | ✅        |

## Citation


If you use ManiSkill3 (versions `mani_skill>=3.0.0`) in your work please cite our [ManiSkill3 paper](https://arxiv.org/abs/2410.00425) as so:

```
@article{taomaniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  author={Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and Yuan Gao and Xuanlin Li and Tongzhou Mu and Nan Xiao and Arnav Gurha and Viswesh Nagaswamy Rajesh and Yong Woo Choi and Yen-Ru Chen and Zhiao Huang and Roberto Calandra and Rui Chen and Shan Luo and Hao Su},
  journal = {Robotics: Science and Systems},
  year={2025},
} 
```

If you use ManiSkill2 (version `mani_skill==0.5.3` or lower) in your work please cite the ManiSkill2 paper as so:
```
@inproceedings{gu2023maniskill2,
  title={ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills},
  author={Gu, Jiayuan and Xiang, Fanbo and Li, Xuanlin and Ling, Zhan and Liu, Xiqiang and Mu, Tongzhou and Tang, Yihe and Tao, Stone and Wei, Xinyue and Yao, Yunchao and Yuan, Xiaodi and Xie, Pengwei and Huang, Zhiao and Chen, Rui and Su, Hao},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

Note that some other assets, algorithms, etc. in ManiSkill are from other sources/research. We try our best to include the correct citation bibtex where possible when introducing the different components provided by ManiSkill.

## License

All rigid body environments in ManiSkill are licensed under fully permissive licenses (e.g., Apache-2.0).

The assets are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
