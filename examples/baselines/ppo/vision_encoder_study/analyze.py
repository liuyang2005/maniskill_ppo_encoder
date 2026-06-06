"""结果分析与画图：面向机器人操作的视觉编码器对比研究.

从 runs/ 下的 tensorboard 日志读取指标，按 (env, encoder, aug) 分组并对多个 seed 聚合，
生成作品集所需的对比图表与汇总表。

用法：
    python analyze.py                       # 解析 ./runs，输出到 ./figures
    python analyze.py --runs_dir runs --out figures --env PickCube-v1

输出：
    figures/sample_efficiency.png   样本效率曲线（success vs steps，按编码器）
    figures/final_success.png       最终成功率柱状图（均值±标准差）
    figures/aug_comparison.png      有/无数据增强对比（若数据存在）
    figures/summary.csv             汇总表（成功率/SPS/参数量/显存）

依赖：tensorboard, pandas, matplotlib, seaborn
"""
import argparse
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# run_name 形如: {env}__{encoder}__{aug}__{seed}__{timestamp}
RUN_RE = re.compile(r"^(?P<env>.+?)__(?P<encoder>.+?)__(?P<aug>aug|noaug)__(?P<seed>\d+)__\d+$")

# 主要关注的评测指标（按优先级，取第一个存在的）
SUCCESS_TAGS = ["eval/success_once", "eval/success_at_end", "eval/return"]


def parse_run_name(name):
    m = RUN_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["seed"] = int(d["seed"])
    return d


def load_scalar(acc, tags):
    """返回第一个存在的 tag 的 (steps, values)；都不存在则 None。"""
    available = acc.Tags().get("scalars", [])
    for tag in tags:
        if tag in available:
            events = acc.Scalars(tag)
            steps = np.array([e.step for e in events])
            vals = np.array([e.value for e in events])
            return tag, steps, vals
    return None, None, None


def load_runs(runs_dir, env_filter=None):
    """扫描 runs_dir，返回每个 run 的元信息 + 曲线数据。"""
    runs = []
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"找不到 runs 目录: {runs_dir}")
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name)
        if not os.path.isdir(path):
            continue
        meta = parse_run_name(name)
        if meta is None:
            continue
        if env_filter and meta["env"] != env_filter:
            continue
        acc = EventAccumulator(path)
        try:
            acc.Reload()
        except Exception as e:
            print(f"[跳过] 无法读取 {name}: {e}")
            continue
        succ_tag, steps, vals = load_scalar(acc, SUCCESS_TAGS)
        if succ_tag is None:
            print(f"[跳过] {name} 中找不到评测指标")
            continue
        scalars = acc.Tags().get("scalars", [])

        def last_scalar(tag, default=np.nan):
            return acc.Scalars(tag)[-1].value if tag in scalars else default

        runs.append(dict(
            **meta,
            success_tag=succ_tag,
            steps=steps,
            success=vals,
            final_success=float(vals[-1]),
            sps=last_scalar("charts/SPS"),
            enc_params_M=last_scalar("study/encoder_trainable_params_M"),
        ))
    if not runs:
        raise RuntimeError(f"{runs_dir} 下没有可解析的 run（run_name 需形如 env__encoder__aug__seed__ts）")
    print(f"共加载 {len(runs)} 个 run，评测指标: {runs[0]['success_tag']}")
    return runs


def aggregate_curves(runs, group_keys=("encoder",)):
    """把同组（默认按 encoder）多个 seed 的曲线对齐并聚合为 均值/标准差。"""
    groups = defaultdict(list)
    for r in runs:
        key = tuple(r[k] for k in group_keys)
        groups[key].append(r)
    out = {}
    for key, items in groups.items():
        # 以最短曲线长度对齐
        min_len = min(len(r["steps"]) for r in items)
        steps = items[0]["steps"][:min_len]
        mat = np.stack([r["success"][:min_len] for r in items])  # (n_seed, T)
        out[key] = dict(steps=steps, mean=mat.mean(0), std=mat.std(0), n=len(items))
    return out


def plot_sample_efficiency(runs, out_dir):
    curves = aggregate_curves(runs, ("encoder",))
    plt.figure(figsize=(7, 5))
    palette = sns.color_palette("tab10", len(curves))
    for (color, (key, c)) in zip(palette, sorted(curves.items())):
        enc = key[0]
        plt.plot(c["steps"], c["mean"], label=f"{enc} (n={c['n']})", color=color)
        plt.fill_between(c["steps"], c["mean"] - c["std"], c["mean"] + c["std"],
                         alpha=0.2, color=color)
    plt.xlabel("Environment Steps")
    plt.ylabel("Eval Success Rate")
    plt.title("Sample Efficiency by Vision Encoder")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "sample_efficiency.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"已保存 {p}")


def plot_final_success(runs, out_dir):
    df = pd.DataFrame([dict(encoder=r["encoder"], final_success=r["final_success"]) for r in runs])
    agg = df.groupby("encoder")["final_success"].agg(["mean", "std", "count"]).reset_index()
    plt.figure(figsize=(7, 5))
    order = agg.sort_values("mean", ascending=False)["encoder"].tolist()
    agg = agg.set_index("encoder").loc[order].reset_index()
    plt.bar(agg["encoder"], agg["mean"], yerr=agg["std"].fillna(0), capsize=5,
            color=sns.color_palette("tab10", len(agg)))
    plt.ylabel("Final Eval Success Rate")
    plt.title("Final Success Rate by Vision Encoder (mean ± std)")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "final_success.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"已保存 {p}")


def plot_aug_comparison(runs, out_dir):
    """若同一编码器存在 aug/noaug，对比最终成功率。"""
    df = pd.DataFrame([dict(encoder=r["encoder"], aug=r["aug"],
                            final_success=r["final_success"]) for r in runs])
    has_both = df.groupby("encoder")["aug"].nunique()
    encoders = has_both[has_both > 1].index.tolist()
    if not encoders:
        print("[跳过] 没有同一编码器的 aug/noaug 对比数据")
        return
    sub = df[df["encoder"].isin(encoders)]
    agg = sub.groupby(["encoder", "aug"])["final_success"].mean().reset_index()
    plt.figure(figsize=(7, 5))
    sns.barplot(data=agg, x="encoder", y="final_success", hue="aug")
    plt.ylabel("Final Eval Success Rate")
    plt.title("Effect of Data Augmentation")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "aug_comparison.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"已保存 {p}")


def write_summary(runs, out_dir):
    rows = []
    for r in runs:
        rows.append(dict(env=r["env"], encoder=r["encoder"], aug=r["aug"], seed=r["seed"],
                         final_success=round(r["final_success"], 3),
                         sps=round(r["sps"], 1) if not np.isnan(r["sps"]) else None,
                         enc_params_M=round(r["enc_params_M"], 2) if not np.isnan(r["enc_params_M"]) else None))
    df = pd.DataFrame(rows)
    p = os.path.join(out_dir, "summary.csv")
    df.to_csv(p, index=False)
    print(f"已保存 {p}")
    # 控制台打印分组汇总
    grp = df.groupby(["encoder", "aug"]).agg(
        success_mean=("final_success", "mean"),
        success_std=("final_success", "std"),
        sps_mean=("sps", "mean"),
        params_M=("enc_params_M", "first"),
        n=("seed", "count"),
    ).round(3)
    print("\n===== 分组汇总 =====")
    print(grp.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--env", default=None, help="只分析指定 env 的 run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sns.set_theme(style="whitegrid")
    runs = load_runs(args.runs_dir, env_filter=args.env)

    plot_sample_efficiency(runs, args.out)
    plot_final_success(runs, args.out)
    plot_aug_comparison(runs, args.out)
    write_summary(runs, args.out)
    print("\n分析完成。图表见 ./%s/" % args.out)


if __name__ == "__main__":
    main()
