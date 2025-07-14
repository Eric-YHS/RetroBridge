import pandas as pd
import random
import argparse
from pathlib import Path


def get_wrong_class(current_class, all_classes):
    """
    从所有可能的类别中，随机选择一个不等于当前类别的新类别。

    Args:
        current_class (int): 当前的类别。
        all_classes (set): 所有可能类别的集合。

    Returns:
        int: 一个随机选择的错误类别。
    """
    # 从所有类别中排除当前类别，得到所有“错误”的类别
    wrong_classes = list(all_classes - {current_class})
    # 从错误的类别中随机选择一个
    return random.choice(wrong_classes)


def main(args):
    """
    主函数，用于读取、修改并保存测试集。
    """
    # 检查输入文件是否存在
    if not args.input_file.is_file():
        print(f"错误: 输入文件未找到 -> {args.input_file}")
        return

    print(f"正在读取测试集文件: {args.input_file}")
    df = pd.read_csv(args.input_file)

    # 确定要修改的行数
    num_rows_to_modify = min(args.num_rows, len(df))
    if num_rows_to_modify < args.num_rows:
        print(
            f"警告: 请求修改 {args.num_rows} 行，但文件只有 {len(df)} 行。将修改所有 {len(df)} 行。")

    # 定义所有可能的反应类别 (USPTO-50k 有10个类别, 从1到10)
    all_possible_classes = set(range(1, 11))

    print(f"将修改前 {num_rows_to_modify} 行数据的 'class' 列...")

    # 记录修改前后的对比
    changes_log = []

    # 遍历并修改指定的行
    for i in range(num_rows_to_modify):
        # 使用 .at 进行高效的单点访问
        original_class = df.at[i, 'class']
        new_class = get_wrong_class(original_class, all_possible_classes)
        df.at[i, 'class'] = new_class

        # 每隔25行打印一个修改示例
        if i < 10 or i % 25 == 0:
            changes_log.append(
                f"  - 行 {i+1}: class {original_class} -> {new_class}")

    print("\n修改示例:")
    for log in changes_log:
        print(log)

    # 检查以确保修改成功
    original_classes_subset = pd.read_csv(
        args.input_file, nrows=num_rows_to_modify)['class']
    modified_classes_subset = df.head(num_rows_to_modify)['class']

    num_changed = (original_classes_subset != modified_classes_subset).sum()
    print(f"\n验证: 成功修改了 {num_changed} / {num_rows_to_modify} 行的类别。")

    # 保存到新文件
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f"\n修改后的测试集已保存到: {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="修改测试集CSV文件中指定行数的'class'列。")

    parser.add_argument(
        '--input-file',
        type=Path,
        default='data/uspto50k/raw/uspto50k_test_real.csv',
        help="输入的测试集CSV文件路径 (默认: data/uspto50k/raw/uspto50k_test_real.csv)"
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default='data/uspto50k/raw/uspto50k_test.csv',
        help="输出的修改后CSV文件路径 (默认: data/uspto50k/raw/uspto50k_test.csv)"
    )
    parser.add_argument(
        '--num-rows',
        type=int,
        default=150,
        help="要修改的行数 (从文件开头算起，默认: 150)"
    )

    args = parser.parse_args()
    main(args)
