#!/bin/bash
# 实验矩阵：面向机器人操作的视觉编码器对比研究
# 用法：
#   bash run_experiments.sh smoke    # 快速验证流程（PushCube，短训，单 seed）
#   bash run_experiments.sh main     # 主实验：编码器对比（PushCube，多 seed）
#   bash run_experiments.sh aug      # 数据增强对比（最佳编码器 有/无 增强）
#
# 所有 run 顺序执行（避免显存竞争），日志写到 runs/ 下，供 analyze.py 解析。
#
# 注：主任务选用 PushCube-v1（推方块）。它比 PickCube（抓取+抬起）简单得多，
#     约 1M 步即可学会并出清晰对比曲线，适合小算力作品集；PickCube-RGB 需约 10M 步。

set -e
PY=/home/leserein/code/maniskill/venv/bin/python
MODE=${1:-smoke}

# [STUDY] 减少 CUDA 显存碎片，降低 OOM 概率（6GB 卡上很关键）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 公共超参（适配 6GB 显存） ----
# CAM 可用环境变量覆盖，如:  CAM=64 bash run_experiments.sh main
# 实测：cam=48 + num_envs=192 训练峰值约 5.0GB（含数据增强），留 ~0.8GB 余量，稳定不 OOM。
# （注：256 envs 虽能跑，但 +aug 时峰值逼近 5.5GB 几乎无余量，易偶发 OOM，故降到 192。）
# 注意 smallvit 需 cam 能被 patch_size=8 整除：48 OK。
CAM=${CAM:-48}
# 并行环境数：192 在所有编码器（含 resnet18 / +aug）下均有安全余量
ENVS=${ENVS:-192}
ENVS_HEAVY=${ENVS_HEAVY:-192}
COMMON="--cam_size=$CAM --num_steps=50 --num_minibatches=8 --update_epochs=8 \
        --num_eval_envs=8 --num_eval_steps=50 --eval_freq=10 --no-capture_video"

run () {
  # $1=env $2=encoder $3=seed $4=timesteps $5=num_envs $6=extra_flags
  echo ">>> env=$1 encoder=$2 seed=$3 steps=$4 envs=$5 cam=$CAM extra=$6"
  $PY ppo_rgb_encoders.py --env_id=$1 --encoder=$2 --seed=$3 \
      --total_timesteps=$4 --num_envs=$5 $COMMON $6
}

case $MODE in
  smoke)
    # 快速跑通流程，确认日志与 analyze.py 正常（约几分钟）
    run PushCube-v1 nature 1 300000 $ENVS ""
    run PushCube-v1 impala 1 300000 $ENVS ""
    ;;

  main)
    # 主实验：4 个编码器 × 3 seed，PushCube（约 1.5M 步足以学会并分出差异）
    ENV=PushCube-v1
    STEPS=1500000
    for SEED in 1 2 3; do
      run $ENV nature   $SEED $STEPS $ENVS ""
      run $ENV impala   $SEED $STEPS $ENVS ""
      run $ENV smallvit $SEED $STEPS $ENVS ""
      run $ENV resnet18 $SEED $STEPS $ENVS_HEAVY "--freeze_encoder"
    done
    ;;

  aug)
    # 数据增强对比：在 impala 上对比 有/无 增强 × 3 seed
    ENV=PushCube-v1
    STEPS=1500000
    for SEED in 1 2 3; do
      run $ENV impala $SEED $STEPS $ENVS ""
      run $ENV impala $SEED $STEPS $ENVS "--use_aug"
    done
    ;;

  *)
    echo "未知模式: $MODE (可选: smoke | main | aug)"; exit 1;;
esac

echo "全部实验完成。运行 'python analyze.py' 生成对比图表。"
