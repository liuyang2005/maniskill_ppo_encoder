#!/bin/bash
# ViT 消融对照实验（单 seed）——补齐"救活 ViT"的证据链
#
# 目的：与已在跑的 convstem_vit+warmup 配对，证明是"ConvStem + warmup 的组合"救活了 ViT。
# 两个关键对照（参数与 convstem+warmup 完全对齐，只差受控变量）：
#   A. convstem_vit  无warmup   ← 看 ConvStem 单独够不够
#   B. smallvit      有warmup   ← 看 warmup 单独够不够（无 ConvStem）
#   （convstem_vit + warmup 已在另一进程跑，本脚本不重复）
#
# 用法（等当前 convstem+warmup 跑完、显存释放后再运行）：
#   bash run_vit_ablation.sh
#
# 顺序执行，一次一个，避免显存冲突。
# 注意：run 目录用自动命名（带时间戳）。跑完后用 record_ablation.txt 对照哪个目录是哪个配置，
#       或直接看本脚本打印的 ">>> 开始..." 行 + 目录创建时间一一对应。

set -e
PY=/home/leserein/code/maniskill/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 与当前 convstem_vit+warmup 完全一致的公共参数（保证可比）
COMMON="--env_id=PushCube-v1 --seed=1 --cam_size=48 --num_envs=128 \
        --num_steps=50 --num_minibatches=8 --update_epochs=8 \
        --num_eval_steps=50 --total_timesteps=1500000 --eval_freq=10 \
        --num_eval_envs=8 --no-capture_video --no-save_model"

echo ">>> 开始 [对照A] convstem_vit 无 warmup  $(date 2>/dev/null)"
$PY ppo_rgb_encoders.py --encoder=convstem_vit --warmup_iters=0 $COMMON

echo ">>> 开始 [对照B] smallvit + warmup15  $(date 2>/dev/null)"
$PY ppo_rgb_encoders.py --encoder=smallvit --warmup_iters=15 $COMMON

echo ">>> 两个对照跑完。消融阶梯："
echo "    smallvit基线=0 | [A]convstem无warmup | [B]smallvit+warmup | convstem+warmup(已跑)"
echo ">>> 跑完告诉我，我帮你读出每个配置的 success 曲线并对比。"
