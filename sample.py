# 文件路径: sample.py (最终修复版)

import argparse
import os
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm
import math
import warnings

# ---- 核心依赖 ----
from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic
from src.analysis.rdkit_functions import build_molecule
from src.analysis.visualization import MolecularVisualization
from src.features.extra_features import DummyExtraFeatures, ExtraFeatures
from src.features.extra_features_molecular import ExtraMolecularFeatures
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.metrics.sampling_metrics import SamplingMolecularMetrics
from src.frameworks.discrete_diffusion import DiscreteDiffusion
from src.frameworks.markov_bridge import MarkovBridge
from src.frameworks.one_shot_model import OneShotModel
from src.data.retrobridge_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos

# 尝试导入 PyG 的 Batch 类，用于高效的数据批处理
try:
    from torch_geometric.data import Batch
    PYG_BATCH_AVAILABLE = True
except ImportError:
    PYG_BATCH_AVAILABLE = False


def main(args):
    torch_device = 'cuda:0' if args.device == 'gpu' else 'cpu'
    data_root = os.path.join(args.data, args.dataset)
    checkpoint_name = args.checkpoint.split('/')[-1].replace('.ckpt', '')

    output_dir = os.path.join(args.samples, f'{args.dataset}_{args.mode}')
    table_name = f'{checkpoint_name}_T={args.n_steps}_n={args.n_samples}_seed={args.sampling_seed}.csv'
    table_path = os.path.join(output_dir, table_name)

    # 限制处理前500条数据 (根据您的需求)
    max_to_process = 500

    # 检查并跳过已完成的进度
    skip_first_n = 0
    if os.path.exists(table_path):
        try:
            # batch_size 从 config 中读取，决定了并行处理的分子数
            molecules_per_batch = args.batch_size 
            with open(table_path, 'r') as f:
                num_lines = sum(1 for line in f) - 1 # 减去表头
            if num_lines > 0 and args.n_samples > 0:
                completed_data_points = num_lines // args.n_samples
                if molecules_per_batch > 0:
                    skip_first_n = completed_data_points // molecules_per_batch
        except Exception as e:
            print(f"警告：无法读取之前的进度文件 '{table_path}'。将从头开始。错误: {e}")
            skip_first_n = 0

    print(f'将跳过之前已运行的 {skip_first_n} 个批次。\n')

    os.makedirs(output_dir, exist_ok=True)
    print(f'Samples will be saved to {table_path}')

    # 在脚本开始时创建带表头的空文件，避免后续追加时重复写入表头
    if not os.path.exists(table_path):
        try:
            header_cols = ['product', 'pred', 'true', 'score', 'true_n_dummy_nodes', 'sampled_n_dummy_nodes', 'nll', 'ell']
            pd.DataFrame(columns=header_cols).to_csv(table_path, index=False)
            print(f"已成功创建空的输出文件：{table_path}")
        except Exception as e:
            print(f"警告：无法创建初始文件，请检查路径和权限。错误: {e}")

    # 确定模型类
    model_map = {
        'DiGress': DiscreteDiffusion,
        'OneShot': OneShotModel,
        'RetroBridge': MarkovBridge
    }
    model_class = model_map.get(args.model)
    if model_class is None:
        raise NotImplementedError(f"Model '{args.model}' not recognized.")
    print('Model class:', model_class)

    # 1. 创建 DataModule
    # batch_size 在这里直接决定了并行处理的分子数量
    datamodule = RetroBridgeDataModule(
        data_root=data_root,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False,
        extra_nodes=args.extra_nodes,
        evaluation=True,
        swap=args.swap,
    )

    # 2. 创建 DatasetInfos
    dataset_infos = datamodule.dataset_infos

    # 3. 创建 Extra Features 模块
    extra_features_module = (
        ExtraFeatures(args.extra_features, dataset_info=dataset_infos)
        if hasattr(args, 'extra_features') and args.extra_features not in [None, 'none']
        else DummyExtraFeatures()
    )
    domain_features_module = (
        ExtraMolecularFeatures(dataset_infos=dataset_infos)
        if hasattr(args, 'extra_molecular_features') and args.extra_molecular_features
        else DummyExtraFeatures()
    )

    # 4. 计算模型的输入/输出维度
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features_module,
        domain_features=domain_features_module,
        use_context=args.use_context,
    )

    # 5. 创建其他 metrics 和 visualization 工具
    train_metrics_module = TrainMolecularMetricsDiscrete(dataset_infos)
    train_smiles_list = datamodule.train_smiles if hasattr(datamodule, 'train_smiles') else []
    sampling_metrics_module = SamplingMolecularMetrics(dataset_infos, train_smiles_list)
    visualization_tools = MolecularVisualization(dataset_infos)

    # 6. 加载模型
    print("Loading model from checkpoint with dependencies...")
    model = model_class.load_from_checkpoint(
        args.checkpoint,
        map_location=torch_device,
        dataset_infos=dataset_infos,
        train_metrics=train_metrics_module,
        sampling_metrics=sampling_metrics_module,
        visualization_tools=visualization_tools,
        extra_features=extra_features_module,
        domain_features=domain_features_module
    )
    print("Model loaded successfully.")

    set_deterministic(args.sampling_seed)
    model.eval().to(torch_device)
    model.visualization_tools = None  # 禁用可视化以加速
    if args.n_steps is not None:
        model.T = args.n_steps

    # =================================================================================
    # ---- 全新的“深度并行”采样和写入逻辑 ----
    # =================================================================================
    molecules_per_batch = datamodule.batch_size 
    print(f"\n模式：深度并行采样。每个批次将并行处理 {molecules_per_batch} 个分子。")
    print(f"每个分子将采样 {args.n_samples} 次。")
    print(f"因此，每次模型调用将生成 {molecules_per_batch * args.n_samples} 个样本。")

    dataloader = datamodule.test_dataloader() if args.mode == 'test' else datamodule.val_dataloader()
    
    processed_count = 0
    pbar = tqdm(dataloader, desc="处理分子批次")

    for i, data in enumerate(pbar):
        if i < skip_first_n:
            processed_count += data.ptr.numel() - 1 if hasattr(data, 'ptr') else len(data.batch.unique())
            continue
        
        if processed_count >= max_to_process:
            print(f"\n已处理 {processed_count} 条数据，达到 {max_to_process} 的限制，停止采样。")
            break

        bs = data.ptr.numel() - 1 if hasattr(data, 'ptr') else len(data.batch.unique())
        if bs == 0: continue
        
        if processed_count + bs > max_to_process:
            print(f"警告：当前批次大小 ({bs}) 将使总数超过处理上限 ({max_to_process})。正在跳过此批次。")
            continue

        pbar.set_description(f"处理批次 {i+1}/{len(dataloader)} (已处理 {processed_count} 分子)")
        
        data = data.to(torch_device)
        
        # --- 关键步骤：复制数据以进行深度并行采样 ---
        if PYG_BATCH_AVAILABLE:
            data_list = data.to_data_list()
            repeated_data_list = [item for item in data_list for _ in range(args.n_samples)]
            large_batch_data = Batch.from_data_list(repeated_data_list)
        else: # 兼容旧版 PyG
            large_batch_data = data.repeat(args.n_samples)


        # --- 执行一次性的、大规模的采样 ---
        pred_molecule_list, true_molecule_list, products_list, scores, nlls, ells = model.sample_batch(
            data=large_batch_data,
            batch_id=i,
            batch_size=bs * args.n_samples,
            save_final=0, keep_chain=0, number_chain_steps_to_save=1,
            sample_idx=0,
            save_true_reactants=True,
            use_one_hot=args.use_one_hot if hasattr(args, 'use_one_hot') else False,
        )

        # --- 处理和写入这整个大批次的结果 ---
        df_rows = []
        for mol_idx in range(bs):
            true_mol_data = true_molecule_list[mol_idx * args.n_samples]
            product_mol_data = products_list[mol_idx * args.n_samples]

            true_mol_obj, true_n_dummy = build_molecule(
                true_mol_data[0], true_mol_data[1], dataset_infos.atom_decoder, return_n_dummy_atoms=True
            )
            true_smi = Chem.MolToSmiles(true_mol_obj) if true_mol_obj else None
            product_mol_obj = build_molecule(product_mol_data[0], product_mol_data[1], dataset_infos.atom_decoder)
            product_smi = Chem.MolToSmiles(product_mol_obj) if product_mol_obj else None

            for sample_j in range(args.n_samples):
                current_idx = mol_idx * args.n_samples + sample_j
                
                pred_mol, pred_score, nll, ell = (
                    pred_molecule_list[current_idx], scores[current_idx], nlls[current_idx], ells[current_idx]
                )
                
                pred_mol_obj, n_dummy = build_molecule(
                    pred_mol[0], pred_mol[1], dataset_infos.atom_decoder, return_n_dummy_atoms=True
                )
                pred_smi = Chem.MolToSmiles(pred_mol_obj) if pred_mol_obj else None
                
                df_rows.append({
                    'product': product_smi,
                    'pred': pred_smi,
                    'true': true_smi,
                    'score': pred_score,
                    'true_n_dummy_nodes': RetroBridgeDatasetInfos.max_n_dummy_nodes - true_n_dummy,
                    'sampled_n_dummy_nodes': RetroBridgeDatasetInfos.max_n_dummy_nodes - n_dummy,
                    'nll': nll,
                    'ell': ell,
                })

        # 创建DataFrame并追加写入CSV
        current_batch_table = pd.DataFrame(df_rows)
        current_batch_table.to_csv(table_path, index=False, mode='a', header=False)
        print(f"批次 {i+1} 完成，已将 {len(current_batch_table)} 条记录写入CSV。")
        
        processed_count += bs

    print("\n所有采样任务完成。")


if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'), required=True, help="Path to the YAML config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument('--samples', type=str, required=True, help="Directory to save the output CSV files.")
    parser.add_argument('--model', type=str, required=True, choices=['DiGress', 'OneShot', 'RetroBridge'], help="Name of the model to use.")
    parser.add_argument('--mode', type=str, required=True, choices=['val', 'test'], help="Dataset split to use for sampling.")
    parser.add_argument('--n_samples', type=int, required=True, help="Number of samples to generate per input molecule.")
    parser.add_argument('--n_steps', type=int, default=None, help="Override the number of diffusion steps.")
    parser.add_argument('--sampling_seed', type=int, default=42, help="Random seed for sampling.")
    # sampling_batch_size 已被废弃，其功能由 config 文件中的 batch_size 代替
    # parser.add_argument('--sampling_batch_size', type=int, default=None, help='DEPRECATED. Use batch_size in config file.')
    parser.add_argument('--use_one_hot', action='store_true', default=False, help="Use one-hot encoding for features.")

    # 解析参数
    args_from_cli = parser.parse_args()
    args = parse_yaml_config(args_from_cli)
    main(args=args)