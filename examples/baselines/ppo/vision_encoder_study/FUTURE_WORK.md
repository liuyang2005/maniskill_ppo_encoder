# 未来工作与待办（Future Work）

本文件记录项目的后续改进方向，按优先级排列。

## 已完成
- [x] 编码器可插拔的公平对比框架（nature / impala / smallvit / resnet18）
- [x] DrQ 数据增强模块
- [x] 编码器横向对比（PushCube，单 seed）
- [x] 数据增强对比实验
- [x] 发现 ViT 失效现象
- [x] 提出 ConvStem 改进 + LR warmup
- [x] ViT 救活的消融实验（证明 ConvStem 与 warmup 协同必要）

## 待办（按优先级）

### 1. 补充多 seed（提升统计严谨性）
当前结果为单 seed。计划对主对比与消融各补至 3 seed，报告均值 ± 标准差，确认结论的稳健性。

### 2. 更难任务上的泛化验证
当前主任务 PushCube 较简单（CNN 可达满分）。计划在 PickCube-v1（抓取+抬起）或 StackCube-v1（叠方块）上验证：
- 编码器对比的差异是否在更难任务上更显著
- ConvStem + warmup 的改进是否同样有效（泛化性）

### 3. ViT 失效的更深入诊断
计划记录训练过程中的注意力熵、梯度范数、特征坍缩程度，从训练动态层面刻画 ViT 失效的机理，为改进提供更直接的证据。

### 4. 改进的进一步消融
- 不同 patch_size（已实现 smallvit_p4 变体）对 ViT 的影响
- ConvStem 层数 / 下采样倍率的影响
- warmup 步数的敏感性分析

## 技术参考
- Xiao et al. 2021, "Early Convolutions Help Transformers See Better"（ConvStem 依据）
- Dosovitskiy et al. 2021, Vision Transformer
- Kostrikov et al. 2021, DrQ 数据增强
