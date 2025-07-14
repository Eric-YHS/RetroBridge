# 假设 src 目录在你的 Python路径中，或者 evaluate_baseline.py 与 RetroBridge 目录同级
# 如果 evaluate_baseline.py 在 RetroBridge 仓库的根目录下，可以这样添加 src 路径：
import sys
sys.path.append('./src')  # 或者根据你的实际路径调整


from metrics.eval_csv_helpers import canonicalize, compute_confidence, assign_groups, compute_accuracy
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial  # 需要导入 partial


def evaluate_predictions(csv_filepath, samples_per_product=10, top_k_values=[1, 3, 5, 10]):
    """
    评估模型预测结果的准确性。

    参数:
    csv_filepath (str or Path): sample.py 生成的 CSV 文件路径。
    samples_per_product (int): sample.py 中 --n_samples 参数的值，即每个产物生成的候选反应物数量。
    top_k_values (list of int): 需要计算的 Top-K 准确率的 K 值列表。
    """
    print(f"正在评估文件: {csv_filepath}")
    df = pd.read_csv(csv_filepath)
    print(f"原始数据行数: {len(df)}")

    # 1. 分组：根据每个产物生成的样本数进行分组
    # assign_groups 函数需要 'from_file' 列，如果 sample.py 的输出没有，我们需要模拟一个
    if 'from_file' not in df.columns:
        df['from_file'] = 'single_file_run'  # 或者使用 csv_filepath.name
    df = assign_groups(df, samples_per_product_per_file=samples_per_product)
    print(f"分组后的数据行数 (应与原始数据一致): {len(df)}")
    print(f"唯一的分组数量 (产物数量): {df['group'].nunique()}")

    # 2. 处理论文中提到的 'C' 产物/反应物的特殊情况
    # 这确保了如果真实反应是 C -> C 这种简单情况，不会因为 SMILES 相同而被错误地标记为不匹配
    # (如果 'Placeholder' 不是一个有效的 SMILES，它在 canonicalize 后会变成 NaN，这可能需要后续处理)
    # 或者，如果 'C' -> 'C' 的情况在你的数据集中不重要或不存在，可以注释掉这部分
    df.loc[(df['product'] == 'C') & (df['true'] == 'C'),
           'true'] = 'Placeholder_True_Reactant'
    # 注意：如果 'Placeholder_True_Reactant' 也是 'C'，那么这一步没有改变。
    # 关键是确保真实的 'C' 反应物在比较时能被正确处理。
    # 如果你的真实反应物中没有 'C'，或者 'C' 产物对应的真实反应物不是 'C'，则此步骤可能不需要。
    # 为了安全起见，我们保留它，但要注意其影响。

    # 3. 计算置信度 (如果 sample.py 的输出包含 'score' 列，并且你想用它)
    # compute_confidence 是基于 'pred' 列的出现频率来计算置信度的。
    # 如果你的 sample.py 输出的 'score' 列是模型直接给出的某种置信度或概率，
    # 你可能想直接使用那个 'score'，或者结合它。
    # 这里的 compute_confidence 是基于重复预测的频率。
    df_processed = compute_confidence(df)
    print(f"计算置信度后的数据行数: {len(df_processed)}")

    # 4. 标准化 SMILES 字符串
    # 这一步非常重要，确保化学上等价但SMILES字符串不同的分子被视为相同
    print("正在标准化 SMILES...")
    # 需要标准化的列：产物、真实反应物、预测反应物
    # 'pred_product' 列在这里不存在，因为我们跳过了 round_trip.py
    columns_to_canonicalize = ['product', 'true', 'pred']
    for key in columns_to_canonicalize:
        if key in df_processed.columns:
            print(f"  标准化列: {key}")
            df_processed[key] = df_processed[key].apply(canonicalize)
        else:
            print(f"  警告: 列 {key} 不在 DataFrame 中，跳过标准化。")

    # 检查是否有因为标准化失败而产生的 NaN 值
    for key in columns_to_canonicalize:
        if key in df_processed.columns:
            nan_count = df_processed[key].isna().sum()
            if nan_count > 0:
                print(f"  警告: 列 {key} 在标准化后包含 {nan_count} 个 NaN 值。")

    # 5. 计算准确率
    # 使用 'confidence' (通过 compute_confidence 计算得到) 作为打分依据
    # 如果你的 'score' 列更有意义，可以替换 scoring 函数
    print(f"正在计算 Top-K 准确率 (K={top_k_values})...")
    # 确保 'confidence' 列存在
    if 'confidence' not in df_processed.columns:
        print("错误: 'confidence' 列不存在。compute_confidence 可能未正确运行或输入数据缺少必要列。")
        # 可以选择提供一个默认打分，例如，如果 'score' 列存在
        if 'score' in df_processed.columns:
            print("将尝试使用 'score' 列进行打分。")
            def scoring_function(
                df_group): return df_group['score']  # 直接使用原始score
        else:
            print("没有可用的打分列，准确率计算可能不准确。")
            scoring_function = None  # 无打分，依赖于原始顺序
    else:
        # 使用基于重复预测频率的置信度
        def scoring_function(df_group): return np.log(
            df_group['confidence'] + 1e-9)  # 加一个小常数避免log(0)

    accuracy_results = compute_accuracy(
        df_processed, top=top_k_values, scoring=scoring_function, verbose=True)

    print("\n--- 评估结果 ---")
    print(accuracy_results)
    print("------------------\n")

    return accuracy_results


if __name__ == "__main__":
    # --- 配置你的参数 ---
    # 替换为你实际的 sample.py 输出文件路径
    CSV_FILE_PATH = Path(
        'samples_new_model/uspto50k_test/epoch=359_top_5_accuracy=0.000_T=500_n=100_seed=42.csv')
    # 这个值必须与你运行 sample.py 时使用的 --n_samples 参数一致
    SAMPLES_PER_PRODUCT = 100
    # 你想看的 Top-K 值
    TOP_K_VALUES = [1, 3, 5, 10]
    # --- 配置结束 ---

    if not CSV_FILE_PATH.exists():
        print(f"错误: CSV 文件未找到: {CSV_FILE_PATH}")
        print("请确保路径正确，并且已经运行了 sample.py 生成了该文件。")
    else:
        evaluate_predictions(CSV_FILE_PATH, SAMPLES_PER_PRODUCT, TOP_K_VALUES)
