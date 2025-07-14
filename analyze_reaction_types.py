#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析USPTO-50数据集中每种反应物对应的反应类型数量
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns


def extract_reactants_from_reaction(reaction_str):
    """
    从反应字符串中提取反应物
    反应格式: reactants>reagents>production
    """
    try:
        parts = reaction_str.split('>')
        if len(parts) >= 3:
            reactants_str = parts[0]
            # 分离多个反应物（用'.'分隔）
            reactants = reactants_str.split('.')
            return [reactant.strip() for reactant in reactants if reactant.strip()]
        return []
    except:
        return []


def extract_products_from_reaction(reaction_str):
    """
    从反应字符串中提取产物
    反应格式: reactants>reagents>production
    """
    try:
        parts = reaction_str.split('>')
        if len(parts) >= 3:
            products_str = parts[2]
            # 分离多个产物（用'.'分隔）
            products = products_str.split('.')
            return [product.strip() for product in products if product.strip()]
        return []
    except:
        return []


def main():
    # 读取数据
    print("正在读取数据...")
    df = pd.read_csv(
        '/GPUFS/nsccgz_ywang_wzh/huangjh/RetroBridge-main/upsto50k_all.csv')

    print(f"数据集大小: {len(df)} 条记录")
    print(f"反应类型数量: {df['class'].nunique()}")
    print(f"反应类型分布:")
    print(df['class'].value_counts().sort_index())

    # 创建反应物到反应类型的映射
    reactant_to_classes = defaultdict(set)
    # 创建反应物组合到反应类型的映射
    reactant_combo_to_classes = defaultdict(set)
    # 创建产物到反应类型的映射
    product_to_classes = defaultdict(set)

    print("\n正在分析反应物、产物和反应类型...")

    # 用于统计分析
    empty_reactants_count = 0
    empty_products_count = 0
    total_processed = 0
    combo_to_reactions = defaultdict(list)  # 记录每个组合对应的反应ID

    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"处理进度: {idx}/{len(df)}")

        reaction_class = row['class']
        reaction_str = row['reactants>reagents>production']
        reaction_id = row['id']

        # 提取反应物
        reactants = extract_reactants_from_reaction(reaction_str)
        # 提取产物
        products = extract_products_from_reaction(reaction_str)
        total_processed += 1

        # 为每个反应物添加对应的反应类型
        for reactant in reactants:
            reactant_to_classes[reactant].add(reaction_class)

        # 为每个产物添加对应的反应类型
        for product in products:
            product_to_classes[product].add(reaction_class)

        # 为反应物组合添加对应的反应类型
        if len(reactants) > 0:
            # 对反应物进行排序以确保组合的一致性
            reactant_combo = tuple(sorted(reactants))
            reactant_combo_to_classes[reactant_combo].add(reaction_class)
            combo_to_reactions[reactant_combo].append(reaction_id)
        else:
            empty_reactants_count += 1

        if len(products) == 0:
            empty_products_count += 1

    print(f"\n总共处理 {total_processed} 条记录")
    print(f"没有反应物的记录数: {empty_reactants_count}")
    print(f"没有产物的记录数: {empty_products_count}")
    print(f"总共发现 {len(reactant_to_classes)} 个不同的反应物")
    print(f"总共发现 {len(product_to_classes)} 个不同的产物")
    print(f"总共发现 {len(reactant_combo_to_classes)} 个不同的反应物组合")

    # 分析重复组合
    duplicate_combos = {combo: reactions for combo,
                        reactions in combo_to_reactions.items() if len(reactions) > 1}
    print(f"有重复的反应物组合数: {len(duplicate_combos)}")

    total_duplicate_reactions = sum(len(reactions)
                                    for reactions in duplicate_combos.values())
    print(f"涉及重复组合的反应总数: {total_duplicate_reactions}")
    print(f"期望的组合数 (如果没有重复): {total_processed - empty_reactants_count}")
    print(f"实际的组合数: {len(reactant_combo_to_classes)}")
    print(
        f"节省的组合数 (由于重复): {total_processed - empty_reactants_count - len(reactant_combo_to_classes)}")

    # 统计每个反应物对应的反应类型数量
    reactant_class_counts = {}
    for reactant, classes in reactant_to_classes.items():
        reactant_class_counts[reactant] = len(classes)

    # 统计每个产物对应的反应类型数量
    product_class_counts = {}
    for product, classes in product_to_classes.items():
        product_class_counts[product] = len(classes)

    # 统计每个反应物组合对应的反应类型数量
    reactant_combo_class_counts = {}
    for combo, classes in reactant_combo_to_classes.items():
        reactant_combo_class_counts[combo] = len(classes)

    # 统计有n种反应类型的反应物数量
    class_count_distribution = Counter(reactant_class_counts.values())

    # 统计有n种反应类型的产物数量
    product_class_count_distribution = Counter(product_class_counts.values())

    # 统计有n种反应类型的反应物组合数量
    combo_class_count_distribution = Counter(
        reactant_combo_class_counts.values())

    print("\n=== 单个反应物统计结果 ===")
    print("每种反应物对应的反应类型数量分布:")

    for i in range(1, 11):  # 统计1-10种反应类型
        count = class_count_distribution.get(i, 0)
        print(f"有{i}种反应类型的反应物数: {count}")

    # 检查是否有超过10种反应类型的反应物
    max_classes = max(reactant_class_counts.values())
    if max_classes > 10:
        print(f"\n注意: 发现有反应物最多对应 {max_classes} 种反应类型")
        for i in range(11, max_classes + 1):
            count = class_count_distribution.get(i, 0)
            if count > 0:
                print(f"有{i}种反应类型的反应物数: {count}")

    print("\n=== 单个产物统计结果 ===")
    print("每种产物对应的反应类型数量分布:")

    for i in range(1, 11):  # 统计1-10种反应类型
        count = product_class_count_distribution.get(i, 0)
        print(f"有{i}种反应类型的产物数: {count}")

    # 检查是否有超过10种反应类型的产物
    max_product_classes = max(product_class_counts.values())
    if max_product_classes > 10:
        print(f"\n注意: 发现有产物最多对应 {max_product_classes} 种反应类型")
        for i in range(11, max_product_classes + 1):
            count = product_class_count_distribution.get(i, 0)
            if count > 0:
                print(f"有{i}种反应类型的产物数: {count}")

    print("\n=== 反应物组合统计结果 ===")
    print("每种反应物组合对应的反应类型数量分布:")

    for i in range(1, 11):  # 统计1-10种反应类型
        count = combo_class_count_distribution.get(i, 0)
        print(f"有{i}种反应类型的反应物组合数: {count}")

    # 检查是否有超过10种反应类型的反应物组合
    max_combo_classes = max(reactant_combo_class_counts.values())
    if max_combo_classes > 10:
        print(f"\n注意: 发现有反应物组合最多对应 {max_combo_classes} 种反应类型")
        for i in range(11, max_combo_classes + 1):
            count = combo_class_count_distribution.get(i, 0)
            if count > 0:
                print(f"有{i}种反应类型的反应物组合数: {count}")

    # 详细分析
    print(f"\n=== 详细分析 ===")
    print(f"反应物总数: {len(reactant_to_classes)}")
    print(f"产物总数: {len(product_to_classes)}")
    print(f"反应物组合总数: {len(reactant_combo_to_classes)}")
    print(
        f"单个反应物平均对应的反应类型数: {np.mean(list(reactant_class_counts.values())):.2f}")
    print(f"单个反应物中位数: {np.median(list(reactant_class_counts.values())):.2f}")
    print(f"单个反应物最大反应类型数: {max(reactant_class_counts.values())}")
    print(f"单个反应物最小反应类型数: {min(reactant_class_counts.values())}")

    print(
        f"\n单个产物平均对应的反应类型数: {np.mean(list(product_class_counts.values())):.2f}")
    print(f"单个产物中位数: {np.median(list(product_class_counts.values())):.2f}")
    print(f"单个产物最大反应类型数: {max(product_class_counts.values())}")
    print(f"单个产物最小反应类型数: {min(product_class_counts.values())}")

    print(
        f"\n反应物组合平均对应的反应类型数: {np.mean(list(reactant_combo_class_counts.values())):.2f}")
    print(
        f"反应物组合中位数: {np.median(list(reactant_combo_class_counts.values())):.2f}")
    print(f"反应物组合最大反应类型数: {max(reactant_combo_class_counts.values())}")
    print(f"反应物组合最小反应类型数: {min(reactant_combo_class_counts.values())}")

    # 找出反应类型最多的反应物
    print(f"\n反应类型最多的前10个反应物:")
    sorted_reactants = sorted(
        reactant_class_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (reactant, count) in enumerate(sorted_reactants[:10]):
        print(f"{i+1}. 反应类型数: {count}, 反应物: {reactant[:100]}...")
        print(f"   对应的反应类型: {sorted(list(reactant_to_classes[reactant]))}")

    # 找出反应类型最多的产物
    print(f"\n反应类型最多的前10个产物:")
    sorted_products = sorted(
        product_class_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (product, count) in enumerate(sorted_products[:10]):
        print(f"{i+1}. 反应类型数: {count}, 产物: {product[:100]}...")
        print(f"   对应的反应类型: {sorted(list(product_to_classes[product]))}")

    # 找出反应类型最多的反应物组合
    print(f"\n反应类型最多的前10个反应物组合:")
    sorted_combos = sorted(
        reactant_combo_class_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (combo, count) in enumerate(sorted_combos[:10]):
        print(f"{i+1}. 反应类型数: {count}")
        print(f"   反应物组合 ({len(combo)} 个反应物):")
        for j, reactant in enumerate(combo):
            print(f"     {j+1}. {reactant[:80]}...")
        print(f"   对应的反应类型: {sorted(list(reactant_combo_to_classes[combo]))}")
        print(f"   涉及的反应数: {len(combo_to_reactions[combo])}")
        print()

    # 显示一些重复组合的例子
    if duplicate_combos:
        print(f"\n重复最多的前5个反应物组合:")
        sorted_duplicates = sorted(
            duplicate_combos.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (combo, reactions) in enumerate(sorted_duplicates[:5]):
            print(f"{i+1}. 反应物组合 ({len(combo)} 个反应物):")
            for j, reactant in enumerate(combo):
                print(f"     {j+1}. {reactant[:60]}...")
            print(f"   重复次数: {len(reactions)}")
            print(f"   涉及的反应ID: {reactions[:10]}...")  # 只显示前10个
            print(
                f"   对应的反应类型: {sorted(list(reactant_combo_to_classes[combo]))}")
            print()

    # 保存详细结果
    print("\n正在保存详细结果...")

    # 保存反应物-反应类型映射
    with open('/GPUFS/nsccgz_ywang_wzh/huangjh/RetroBridge-main/reactant_class_mapping.txt', 'w', encoding='utf-8') as f:
        f.write("反应物\t反应类型数\t反应类型列表\n")
        for reactant, classes in sorted(reactant_to_classes.items(), key=lambda x: len(x[1]), reverse=True):
            f.write(f"{reactant}\t{len(classes)}\t{sorted(list(classes))}\n")

    # 保存产物-反应类型映射
    with open('/GPUFS/nsccgz_ywang_wzh/huangjh/RetroBridge-main/product_class_mapping.txt', 'w', encoding='utf-8') as f:
        f.write("产物\t反应类型数\t反应类型列表\n")
        for product, classes in sorted(product_to_classes.items(), key=lambda x: len(x[1]), reverse=True):
            f.write(f"{product}\t{len(classes)}\t{sorted(list(classes))}\n")

    # 保存反应物组合-反应类型映射
    with open('/GPUFS/nsccgz_ywang_wzh/huangjh/RetroBridge-main/reactant_combo_class_mapping.txt', 'w', encoding='utf-8') as f:
        f.write("反应物组合\t反应物数量\t反应类型数\t反应类型列表\n")
        for combo, classes in sorted(reactant_combo_to_classes.items(), key=lambda x: len(x[1]), reverse=True):
            combo_str = " + ".join(combo)
            f.write(
                f"{combo_str}\t{len(combo)}\t{len(classes)}\t{sorted(list(classes))}\n")

    # 保存统计结果
    with open('/GPUFS/nsccgz_ywang_wzh/huangjh/RetroBridge-main/reaction_type_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("=== USPTO-50数据集反应类型统计结果 ===\n\n")
        f.write(f"数据集大小: {len(df)} 条记录\n")
        f.write(f"反应物总数: {len(reactant_to_classes)}\n")
        f.write(f"产物总数: {len(product_to_classes)}\n")
        f.write(f"反应物组合总数: {len(reactant_combo_to_classes)}\n")
        f.write(f"反应类型数量: {df['class'].nunique()}\n\n")

        f.write("=== 单个反应物统计 ===\n")
        f.write("每种反应物对应的反应类型数量分布:\n")
        for i in range(1, max_classes + 1):
            count = class_count_distribution.get(i, 0)
            f.write(f"有{i}种反应类型的反应物数: {count}\n")

        f.write(
            f"\n单个反应物平均对应的反应类型数: {np.mean(list(reactant_class_counts.values())):.2f}\n")
        f.write(
            f"单个反应物中位数: {np.median(list(reactant_class_counts.values())):.2f}\n")
        f.write(f"单个反应物最大反应类型数: {max(reactant_class_counts.values())}\n")
        f.write(f"单个反应物最小反应类型数: {min(reactant_class_counts.values())}\n")

        f.write("\n=== 单个产物统计 ===\n")
        f.write("每种产物对应的反应类型数量分布:\n")
        for i in range(1, max_product_classes + 1):
            count = product_class_count_distribution.get(i, 0)
            f.write(f"有{i}种反应类型的产物数: {count}\n")

        f.write(
            f"\n单个产物平均对应的反应类型数: {np.mean(list(product_class_counts.values())):.2f}\n")
        f.write(
            f"单个产物中位数: {np.median(list(product_class_counts.values())):.2f}\n")
        f.write(f"单个产物最大反应类型数: {max(product_class_counts.values())}\n")
        f.write(f"单个产物最小反应类型数: {min(product_class_counts.values())}\n")

        f.write("\n=== 反应物组合统计 ===\n")
        f.write("每种反应物组合对应的反应类型数量分布:\n")
        for i in range(1, max_combo_classes + 1):
            count = combo_class_count_distribution.get(i, 0)
            f.write(f"有{i}种反应类型的反应物组合数: {count}\n")

        f.write(
            f"\n反应物组合平均对应的反应类型数: {np.mean(list(reactant_combo_class_counts.values())):.2f}\n")
        f.write(
            f"反应物组合中位数: {np.median(list(reactant_combo_class_counts.values())):.2f}\n")
        f.write(f"反应物组合最大反应类型数: {max(reactant_combo_class_counts.values())}\n")
        f.write(f"反应物组合最小反应类型数: {min(reactant_combo_class_counts.values())}\n")

    print("分析完成！结果已保存到:")
    print("- reactant_class_mapping.txt: 详细的反应物-反应类型映射")
    print("- product_class_mapping.txt: 详细的产物-反应类型映射")
    print("- reactant_combo_class_mapping.txt: 详细的反应物组合-反应类型映射")
    print("- reaction_type_statistics.txt: 统计结果汇总")


if __name__ == "__main__":
    main()
