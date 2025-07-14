# parallel_sample.py (final version with sample.py-like logic)

from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from rdkit import Chem
from src.features.extra_features_molecular import ExtraMolecularFeatures
from src.features.extra_features import DummyExtraFeatures, ExtraFeatures
from src.metrics.sampling_metrics import DummySamplingMolecularMetrics
from src.metrics.molecular_metrics_discrete import DummyTrainMolecularMetricsDiscrete
from src.frameworks.markov_bridge import MarkovBridge
from src.data.retrobridge_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos
from src.analysis.rdkit_functions import build_molecule
from src.utils import disable_rdkit_logging, set_deterministic
from tqdm import tqdm
import yaml
import multiprocessing
import torch
import numpy as np
import pandas as pd
import os
import argparse
import sys
sys.path.append('src')


def run_sampling_on_device(device_id, all_indices_for_gpu, skip_batches_for_gpu, config):
    """
    在单个GPU上执行采样任务，逻辑与 sample.py 对齐。
    """
    try:
        # 1. 设置环境
        set_deterministic(int(config.seed) + device_id)
        torch_device = f'cuda:{device_id}'
        torch.cuda.set_device(torch_device)
        print(f"[GPU {device_id}]: Starting sampling. Tasks: {len(all_indices_for_gpu)}, Skipping first {skip_batches_for_gpu} batches.")

        # 2. 准备输出路径
        output_dir = os.path.dirname(config.output_file)
        base_name = os.path.basename(config.output_file).replace('.csv', '')
        temp_table_path = os.path.join(
            output_dir, f"{base_name}_part_{device_id}.csv")
        os.makedirs(output_dir, exist_ok=True)
        print(
            f"[GPU {device_id}]: Temporary results will be saved to {temp_table_path}")

        # 3. 准备模型加载依赖
        datamodule = RetroBridgeDataModule(
            data_root=os.path.join(config.data, config.dataset),
            batch_size=config.batch_size, num_workers=0, shuffle=False,
            extra_nodes=config.extra_nodes, evaluation=True, swap=config.swap,
        )
        dataset_infos = RetroBridgeDatasetInfos(datamodule)

        extra_features = DummyExtraFeatures() if not config.extra_features else ExtraFeatures(
            config.extra_features, dataset_infos)
        domain_features = ExtraMolecularFeatures(
            dataset_infos) if config.extra_molecular_features else DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule, extra_features=extra_features, domain_features=domain_features, use_context=config.use_context
        )

        args_for_load = {
            'dataset_infos': dataset_infos,
            'train_metrics': DummyTrainMolecularMetricsDiscrete(), 'sampling_metrics': DummySamplingMolecularMetrics(), 'visualization_tools': None
        }

        # 4. 加载模型
        model = MarkovBridge.load_from_checkpoint(
            config.checkpoint, map_location=torch_device, **args_for_load)
        model.extra_features = extra_features
        model.domain_features = domain_features
        model.eval().to(torch_device)
        model.T = config.diffusion_steps

        # 5. 准备Dataloader
        full_test_dataset = datamodule.test_dataloader().dataset
        subset_for_device = Subset(full_test_dataset, all_indices_for_gpu)
        dataloader = DataLoader(
            dataset=subset_for_device, batch_size=config.batch_size, num_workers=0, shuffle=False)

        # 6. 核心采样循环
        group_size = config.n_samples

        pbar = tqdm(dataloader, desc=f"GPU {device_id}", position=device_id)
        for i, data in enumerate(pbar):
            if i < skip_batches_for_gpu:
                continue

            pbar.set_description(
                f"GPU {device_id} - Batch {i+1}/{len(dataloader)}")
            data = data.to(torch_device)
            bs = len(data.batch.unique())
            batch_groups, ground_truth, input_products = [], [], []

            for sample_idx in range(group_size):
                pred_molecule_list, true_molecule_list, products_list, _, _, _ = model.sample_batch(
                    data=data, batch_id=0, batch_size=bs, save_final=0, keep_chain=0,
                    number_chain_steps_to_save=1, sample_idx=sample_idx,
                    save_true_reactants=True, use_one_hot=False
                )
                batch_groups.append(pred_molecule_list)
                if sample_idx == 0:
                    ground_truth.extend(true_molecule_list)
                    input_products.extend(products_list)

            # --- 结果处理与保存逻辑 ---
            current_batch_results = []
            for mol_idx_in_batch in range(bs):
                true_mol_data = ground_truth[mol_idx_in_batch]
                prod_mol_data = input_products[mol_idx_in_batch]
                pred_mols_data = [bg[mol_idx_in_batch] for bg in batch_groups]

                true_rdkit, _ = build_molecule(
                    true_mol_data[0], true_mol_data[1], dataset_infos.atom_decoder, return_n_dummy_atoms=True)
                true_smi = Chem.MolToSmiles(true_rdkit) if true_rdkit else None

                prod_rdkit = build_molecule(
                    prod_mol_data[0], prod_mol_data[1], dataset_infos.atom_decoder)
                prod_smi = Chem.MolToSmiles(prod_rdkit) if prod_rdkit else None

                for pred_mol_data in pred_mols_data:
                    pred_rdkit, _ = build_molecule(
                        pred_mol_data[0], pred_mol_data[1], dataset_infos.atom_decoder, return_n_dummy_atoms=True)
                    pred_smi = Chem.MolToSmiles(
                        pred_rdkit) if pred_rdkit else None
                    current_batch_results.append(
                        {'product': prod_smi, 'pred': pred_smi, 'true': true_smi})

            # 将当前批次的结果追加写入临时文件
            if current_batch_results:
                batch_df = pd.DataFrame(current_batch_results)
                header = not os.path.exists(temp_table_path)
                batch_df.to_csv(temp_table_path, mode='a',
                                header=header, index=False)

        pbar.close()
        print(f"[GPU {device_id}]: Sampling finished.")

    except Exception as e:
        print(f"[GPU {device_id}]: An error occurred: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Parallel sampling for RetroBridge.")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to YAML config file.")
    # ... 其他参数保持不变 ...
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to model checkpoint .ckpt file.")
    parser.add_argument('--n_gpus', type=int, default=4,
                        help="Number of GPUs to use.")
    parser.add_argument('--start_row', type=int, required=True,
                        help="Starting row index (1-based) from the test set.")
    parser.add_argument('--end_row', type=int, required=True,
                        help="Ending row index (1-based) from the test set.")
    parser.add_argument('--n_samples', type=int, required=True,
                        help="Number of samples to generate per product molecule.")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Path to the final merged output CSV file.")

    args = parser.parse_args()

    # 配置加载
    with open(args.config, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config_args = argparse.Namespace(**config_dict)
    for key, value in vars(args).items():
        setattr(config_args, key, value)

    print("--- Final Configuration ---")
    [print(f'{k: <40} -> {v}') for k, v in vars(config_args).items()]
    print("--------------------------")
    disable_rdkit_logging()

    # 任务划分
    start_idx = args.start_row - 1
    end_idx = args.end_row
    total_indices = list(range(start_idx, end_idx))
    print(
        f"Total data to process: {len(total_indices)} molecules from index {start_idx} to {end_idx-1}.")

    indices_per_gpu = np.array_split(total_indices, args.n_gpus)
    indices_per_gpu = [x.tolist() for x in indices_per_gpu if len(x) > 0]

    # --- 断点续采逻辑 (主进程) ---
    skip_batches_info = []
    output_dir = os.path.dirname(args.output_file)
    base_name = os.path.basename(args.output_file).replace('.csv', '')
    for i in range(len(indices_per_gpu)):
        temp_file = os.path.join(output_dir, f"{base_name}_part_{i}.csv")
        batches_to_skip = 0
        if os.path.exists(temp_file):
            try:
                with open(temp_file, 'r') as f:
                    num_lines = sum(1 for line in f) - 1
                if num_lines > 0 and config_args.n_samples > 0:
                    completed_data_points = num_lines // config_args.n_samples
                    if config_args.batch_size > 0:
                        batches_to_skip = completed_data_points // config_args.batch_size
            except (IOError, pd.errors.EmptyDataError):
                pass
        skip_batches_info.append(batches_to_skip)

    # 启动进程
    processes = []
    for i in range(len(indices_per_gpu)):
        process = multiprocessing.Process(
            target=run_sampling_on_device,
            args=(i, indices_per_gpu[i], skip_batches_info[i], config_args)
        )
        processes.append(process)
        process.start()
        print(
            f"Started process for GPU {i}, handling {len(indices_per_gpu[i])} molecules, skipping {skip_batches_info[i]} batches.")

    # 等待与合并
    for process in processes:
        process.join()

    print("\nAll sampling processes have finished. Merging results...")

    df_list = []
    for i in range(len(indices_per_gpu)):
        temp_file = os.path.join(output_dir, f"{base_name}_part_{i}.csv")
        if os.path.exists(temp_file):
            print(f"Reading temporary file: {temp_file}")
            df_list.append(pd.read_csv(temp_file))
            os.remove(temp_file)
        else:
            print(f"Warning: Temporary file not found: {temp_file}")

    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.drop_duplicates(inplace=True)
        final_df.to_csv(args.output_file, index=False)
        print(
            f"\nSuccessfully merged {len(final_df)} rows into {args.output_file}")
    else:
        print("\nNo results were generated. The output file was not created.")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
