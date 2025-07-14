# calculate_accuracy.py

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 将项目根目录添加到Python路径中，以便可以导入src模块
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from src.metrics.eval_csv_helpers import canonicalize, compute_confidence, compute_accuracy
    from rdkit import Chem
    # 关闭RDKit的冗余日志输出
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError as e:
    print(f"导入模块时出错: {e}")
    print("请确保您在项目的根目录下运行此脚本，并且您的conda环境已激活。")
    sys.exit(1)


def main(input_file: Path, n_samples: int):
    """
    计算给定采样结果CSV文件的Top-K准确率。

    :param input_file: 包含采样结果的CSV文件路径。
    :param n_samples: 每个输入产物分子的采样次数。
    """
    if not input_file.exists():
        print(f"错误: 输入文件不存在 -> {input_file}")
        sys.exit(1)

    print(f"正在从 {input_file} 读取数据...")
    df = pd.read_csv(input_file)
    print(f"数据加载完成，共 {len(df)} 行。")

    # --- 步骤 1: 分组 (关键步骤) ---
    # 根据每个分子的采样次数(n_samples)对数据进行分组。
    # 这是计算准确率的基础。
    print(f"正在按每组 {n_samples} 条样本进行分组...")
    df['group'] = np.arange(len(df)) // n_samples

    # --- 步骤 2: SMILES 标准化 ---
    # 将SMILES字符串转换为唯一的“规范”形式，以确保正确比较。
    print("正在对 'true' 和 'pred' SMILES 进行标准化...")
    df['true'] = df['true'].apply(canonicalize)
    df['pred'] = df['pred'].apply(canonicalize)
    # 移除无法标准化的行（通常是无效分子）
    df.dropna(subset=['true', 'pred'], inplace=True)

    # --- 步骤 3: 计算置信度分数 ---
    # 置信度被定义为在n_samples次采样中，某个特定预测产物出现的频率。
    # 这将作为我们对预测进行排序的依据。
    print("正在计算每个预测的置信度分数...")
    df_processed = compute_confidence(df)

    # --- 步骤 4: 计算并打印 Top-K 准确率 ---
    # 定义要计算的K值
    top_k_values = [1, 3, 5, 10]

    # 定义评分函数。这里使用置信度的对数作为分数，这是一种常见的做法。
    def scoring_function(d): return np.log(d['confidence'])

    print(f"\n正在计算 Top-{top_k_values} 准确率...")
    # `compute_accuracy` 函数会自动处理分组、排序和计算
    results_df = compute_accuracy(
        df_processed, top=top_k_values, scoring=scoring_function, verbose=True)

    print("\n" + "="*30)
    print("      Top-K 准确率结果")
    print("="*30)
    print(results_df)
    print("="*30)

    # 检查是否可以计算"往返"准确率
    if 'pred_product' in df_processed.columns:
        print("\n检测到 'pred_product' 列，同时计算 'Round-trip' 准确率。")
    else:
        print("\n提示: 未检测到 'pred_product' 列。如需计算 'Round-trip' 准确率，")
        print("请先运行 `src/metrics/round_trip.py`。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从采样CSV文件计算Top-K准确率。")
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="指向采样结果的CSV文件路径 (例如: samples/test.csv)。"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="每个输入产物分子的采样次数 (例如: 100)。"
    )
    args = parser.parse_args()

    main(input_file=args.input_file, n_samples=args.n_samples)
