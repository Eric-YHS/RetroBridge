# 文件路径: src/metrics/eval_csv_helpers.py

import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm
from rdkit import Chem


def canonicalize(smi):
    """
    将 SMILES 字符串标准化。
    
    新增功能：在处理前检查输入是否为字符串。如果输入不是字符串
    （例如，从CSV读取的空值变成了 float 类型的 NaN），则直接返回 np.nan，
    以避免将非字符串传递给 RDKit 导致 TypeError。
    """
    # --- 这是核心的修改部分 ---
    if not isinstance(smi, str):
        return np.nan
    # --- 修改结束 ---

    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return np.nan
    return Chem.MolToSmiles(m)


def _assign_groups(df, samples_per_product):
    """辅助函数，为单个文件的数据帧分配组号。"""
    df['group'] = np.arange(len(df)) // samples_per_product
    return df


def assign_groups(df, samples_per_product_per_file=10):
    """
    根据每个产物生成的样本数量，为整个数据帧分配组。
    这假定数据是按产物顺序排列的。
    """
    df = df.groupby('from_file').apply(partial(_assign_groups, samples_per_product=samples_per_product_per_file))
    return df


def compute_confidence(df):
    """
    根据预测反应物在同一组（同一产物）中的出现频率计算置信度。
    """
    counts = df.groupby(['group', 'pred']).size().reset_index(name='count')
    group_size = df.groupby(['group']).size().reset_index(name='group_size')

    # 使用字典映射以提高性能并避免 merge 可能带来的问题
    counts_dict = {(g, p): c for g, p, c in zip(counts['group'], counts['pred'], counts['count'])}
    # 使用 .get() 并提供默认值 0，以防某个 (group, pred) 组合不存在（尽管在这里不太可能）
    df['count'] = df.apply(lambda x: counts_dict.get((x['group'], x['pred']), 0), axis=1)

    size_dict = {g: s for g, s in zip(group_size['group'], group_size['group_size'])}
    df['group_size'] = df.apply(lambda x: size_dict.get(x['group'], 1), axis=1) # 默认值为1以防除零

    df['confidence'] = df['count'] / df['group_size']

    # 完整性检查
    assert (df.groupby(['group', 'pred'])['confidence'].nunique() == 1).all()
    assert (df.groupby(['group'])['group_size'].nunique() == 1).all()

    return df


def get_top_k(df, k, scoring=None):
    """
    对于给定的数据帧（代表一个产物的所有预测），根据打分获取前 k 个唯一的预测。
    """
    if callable(scoring):
        df["_new_score"] = scoring(df)
        scoring = "_new_score"

    if scoring is not None:
        # 按打分降序排序
        df = df.sort_values(by=scoring, ascending=False)
    
    # 去除重复的预测，保留第一个（即分数最高的那个）
    df = df.drop_duplicates(subset='pred')

    return df.head(k)


def compute_accuracy(df, top=[1, 3, 5], scoring=None, verbose=False):
    """
    计算 Top-K 准确率。
    'round_trip' 相关的逻辑被保留，但只在 'pred_product' 列存在时才会被激活。
    """
    round_trip = 'pred_product' in df.columns

    results = {}
    results['Exact match'] = {}

    df['exact_match'] = df['true'] == df['pred']

    if round_trip:
        results['Round-trip coverage'] = {}
        results['Round-trip accuracy'] = {}

        df['round_trip_match'] = df['product'] == df['pred_product']
        df['match'] = df['exact_match'] | df['round_trip_match']

    for k in tqdm(top, desc="Computing Top-K Accuracy"):
        # 对每个组应用 get_top_k 函数
        topk_df = df.groupby(['group']).apply(partial(get_top_k, k=k, scoring=scoring)).reset_index(drop=True)

        # 计算精确匹配准确率
        acc_exact_match = topk_df.groupby('group').exact_match.any().mean()
        results['Exact match'][f'top-{k}'] = acc_exact_match
        if verbose:
            print(f"\nTop-{k}")
            print("Exact match accuracy", acc_exact_match)

        if round_trip:
            # 计算往返覆盖率和准确率
            cov_round_trip = topk_df.groupby('group').match.any().mean()
            acc_round_trip = topk_df.groupby('group').match.mean().mean()

            results['Round-trip coverage'][f'top-{k}'] = cov_round_trip
            results['Round-trip accuracy'][f'top-{k}'] = acc_round_trip

            if verbose:
                print("Round-trip coverage", cov_round_trip)
                print("Round-trip accuracy", acc_round_trip)

    return pd.DataFrame(results).T