#!/bin/bash
# 高配实验脚本（面向 AutoDL RTX 4090 / 24GB 显存）
# 与 run_experiments.sh 的区别：更大 num_envs、更高相机分辨率、可跑更难的 PickCube。
#
# 用法：
#   bash run_experiments_4090.sh smoke         # 快速验证环境（PushCube，单 seed，短训）
#   bash run_experiments_4090.sh main          # 主实验：PushCube 4 编码器 × 3 seed
#   bash run_experiments_4090.sh aug           # 数据增强对比：impala 有/无增强 × 3 seed
#   bash run_experiments_4090.sh pickcube      # 进阶主实验：PickCube（更难，更有说服力）4 编码器 × 3 seed
#
# 所有 run 顺序执行（避免显存竞争）；日志写到 runs/，供 analyze.py 解析。
# 推荐后台跑：  nohup bash run_experiments_4090.sh main > main.log 2>&1 &  然后 tail -f main.log

set -e
PY=python   # AutoDL 上 python 通常就是实例自带的（如需指定可改成绝对路径）
MODE=${1:-smoke}

# 减少 CUDA 显存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 高配超参（24GB 显存）----
# 均可用环境变量覆盖，例如:  ENVS=1024 bash run_experiments_4090.sh main
CAM=${CAM:-64}            # 相机分辨率：4090 上用回 64×64（比 48 质量更好；注意 smallvit 需被 8 整除，64 OK）
ENVS=${ENVS:-512}         # 并行环境数：4090 24GB 可轻松支撑 512
ENVS_HEAVY=${ENVS_HEAVY:-512}   # resnet18 等较重编码器；若偶发 OOM 可降到 256
UPDATE_EPOCHS=${UPDATE_EPOCHS:-8}

COMMON="--cam_size=$CAM --num_steps=50 --num_minibatches=8 --update_epochs=$UPDATE_EPOCHS \
        --num_eval_envs=16 --num_eval_steps=50 --eval_freq=10 --no-capture_video"

run () {
  # $1=env $2=encoder $3=seed $4=timesteps $5=num_envs $6=extra_flags
  echo ">>> env=$1 encoder=$2 seed=$3 steps=$4 envs=$5 cam=$CAM extra=$6"
  $PY ppo_rgb_encoders.py --env_id=$1 --encoder=$2 --seed=$3 \
      --total_timesteps=$4 --num_envs=$5 $COMMON $6
}

case $MODE in
  smoke)
    # 快速验证环境与流程（几分钟）
    run PushCube-v1 nature 1 300000 $ENVS ""
    run PushCube-v1 impala 1 300000 $ENVS ""
    ;;

  main)
    # 主实验：PushCube，4 编码器 × 3 seed
    ENV=PushCube-v1
    STEPS=${STEPS:-1500000}
    for SEED in 1 2 3; do
      run $ENV nature   $SEED $STEPS $ENVS ""
      run $ENV impala   $SEED $STEPS $ENVS ""
      run $ENV smallvit $SEED $STEPS $ENVS ""
      run $ENV resnet18 $SEED $STEPS $ENVS_HEAVY "--freeze_encoder"
    done
    ;;

  aug)
    # 数据增强对比：impala 有/无增强 × 3 seed
    ENV=PushCube-v1
    STEPS=${STEPS:-1500000}
    for SEED in 1 2 3; do
      run $ENV impala $SEED $STEPS $ENVS ""
      run $ENV impala $SEED $STEPS $ENVS "--use_aug"
    done
    ;;

  pickcube)
    # 进阶主实验：PickCube（抓取+抬起，比 PushCube 难，更有说服力）
    # PickCube-RGB 需更多步数才学会，4090 算力足够。默认 5M 步/run。
    ENV=PickCube-v1
    STEPS=${STEPS:-5000000}
    for SEED in 1 2 3; do
      run $ENV nature   $SEED $STEPS $ENVS ""
      run $ENV impala   $SEED $STEPS $ENVS ""
      run $ENV smallvit $SEED $STEPS $ENVS ""
      run $ENV resnet18 $SEED $STEPS $ENVS_HEAVY "--freeze_encoder"
    done
    ;;

  *)
    echo "未知模式: $MODE (可选: smoke | main | aug | pickcube)"; exit 1;;
esac

echo "全部实验完成。运行 'python analyze.py --env <环境名>' 生成对比图表。"
