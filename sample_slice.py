# sample_slice.py (最终修复版 v4)
from pdb import set_trace
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from src.features.extra_features_molecular import ExtraMolecularFeatures
from src.features.extra_features import DummyExtraFeatures, ExtraFeatures
from src.data.abstract_dataset import MolecularDataModule
from src.data.retrobridge_dataset import RetroBridgeDataset, RetroBridgeDatasetInfos
from src.metrics.sampling_metrics import DummySamplingMolecularMetrics
from src.metrics.molecular_metrics_discrete import DummyTrainMolecularMetricsDiscrete
from src.frameworks.markov_bridge import MarkovBridge
from src.analysis.rdkit_functions import build_molecule
from src.utils import disable_rdkit_logging, set_deterministic
import math
import torch
import pandas as pd
import os
import argparse
import sys
sys.path.append('src')


# ### 关键改动: 重新添加缺失的导入 ###


class MinimalDataModuleForInfos(MolecularDataModule):
    def __init__(self, data_root, batch_size, num_workers, extra_nodes, swap):
        super().__init__(batch_size, num_workers, shuffle=False)
        self.data_root = data_root
        self.extra_nodes = extra_nodes
        self.swap = swap

        train_dataset = RetroBridgeDataset(
            stage='train', root=self.data_root, extra_nodes=self.extra_nodes, swap=self.swap)
        val_dataset = RetroBridgeDataset(
            stage='val', root=self.data_root, extra_nodes=self.extra_nodes, swap=self.swap)
        test_dataset = RetroBridgeDataset(
            stage='test', root=self.data_root, extra_nodes=self.extra_nodes, swap=self.swap)

        self.dataloaders = {
            "train": DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers),
            "val": DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers),
            "test": DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
        }
        self.evaluation = False


def main(gpu_id, data_slice, n_samples_per_input, sample_batch_size, checkpoint_path):
    # --- 1. 配置参数 ---
    torch_device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"将在设备 {torch_device} 上运行")

    config = {
        "dataset": "uspto50k",
        "data_root": "datasets",
        "num_workers": 8,
        "extra_nodes": True,
        "swap": False,
        "n_steps": 500,
        "sampling_seed": 42,
        "extra_features": "all",
        "extra_molecular_features": True,
        "use_context": True
    }
    set_deterministic(config["sampling_seed"])

    # --- 2. 设置输出路径 ---
    experiment_name = checkpoint_path.split('/')[-3]
    checkpoint_name = checkpoint_path.split('/')[-1].replace('.ckpt', '')
    slice_name = f"slice_{data_slice.start}_{data_slice.stop}"
    output_dir = os.path.join(
        'samples', experiment_name, f'{config["dataset"]}_70_{slice_name}')
    table_name = f'{checkpoint_name}_T={config["n_steps"]}_n={n_samples_per_input}_seed={config["sampling_seed"]}.csv'

    table_path = os.path.join(output_dir, table_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"采样结果将保存到: {table_path}")

    # --- 3. 准备加载模型所需的依赖项 ---
    print("正在准备加载模型所需的依赖项...")
    data_root_path = os.path.join(config["data_root"], config["dataset"])

    datamodule_for_infos = MinimalDataModuleForInfos(
        data_root=data_root_path, batch_size=sample_batch_size, num_workers=config[
            "num_workers"],
        extra_nodes=config["extra_nodes"], swap=config["swap"]
    )
    print("正在计算/加载数据集统计信息...")
    dataset_infos = RetroBridgeDatasetInfos(datamodule_for_infos)

    extra_features = (
        ExtraFeatures(config["extra_features"], dataset_info=dataset_infos)
        if config["extra_features"] is not None
        else DummyExtraFeatures()
    )
    domain_features = (
        ExtraMolecularFeatures(dataset_infos=dataset_infos)
        if config["extra_molecular_features"]
        else DummyExtraFeatures()
    )

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule_for_infos,
        extra_features=extra_features,
        domain_features=domain_features,
        use_context=config["use_context"]
    )
    print("统计信息和维度准备完成。")

    # --- 4. 加载模型 ---
    print(f"正在从 {checkpoint_path} 加载模型...")
    model = MarkovBridge.load_from_checkpoint(
        checkpoint_path, map_location=torch_device, dataset_infos=dataset_infos,
        train_metrics=DummyTrainMolecularMetricsDiscrete(), sampling_metrics=DummySamplingMolecularMetrics(), visualization_tools=None
    )

    model.extra_features = extra_features
    model.domain_features = domain_features
    model.eval().to(torch_device)
    model.T = config["n_steps"]

    # --- 5. 数据加载和切片 ---
    full_test_dataset = RetroBridgeDataset(
        stage='test', root=data_root_path, extra_nodes=config["extra_nodes"], swap=config["swap"])
    sliced_dataset = full_test_dataset[data_slice]
    print(
        f"已加载并切分数据集。使用 {len(sliced_dataset)} 条数据，从索引 {data_slice.start} 到 {data_slice.stop - 1}")

    # --- 6. 断点续采和文件初始化 ---
    start_molecule_idx = 0
    if os.path.exists(table_path):
        try:
            prev_table = pd.read_csv(table_path)
            if not prev_table.empty:
                num_completed_inputs = prev_table['product'].nunique()
                start_molecule_idx = num_completed_inputs
                print(
                    f"检测到已有结果文件。已完成 {num_completed_inputs} 个输入分子的采样。将从第 {start_molecule_idx} 个分子开始。")
            else:
                print("检测到空的结果文件，将从头开始。")
        except pd.errors.EmptyDataError:
            print(f"检测到空的结果文件，将从头开始。")
    else:
        pd.DataFrame(columns=[
            'product', 'pred', 'true', 'score', 'true_n_dummy_nodes',
            'sampled_n_dummy_nodes', 'nll', 'ell'
        ]).to_csv(table_path, index=False)

    # --- 7. 主采样循环 (按分子写入，并行采样版) ---
    for mol_idx, input_molecule_data in enumerate(tqdm(sliced_dataset, desc="Processing Molecules")):
        if mol_idx < start_molecule_idx:
            continue

        all_samples_for_this_molecule = []
        num_batches = math.ceil(n_samples_per_input / sample_batch_size)

        for batch_num in range(num_batches):
            current_batch_size = min(
                sample_batch_size, n_samples_per_input - (batch_num * sample_batch_size))
            if current_batch_size <= 0:
                continue

            batch_data_list = [input_molecule_data.clone()
                               for _ in range(current_batch_size)]

            # 使用 DataLoader collate 功能来创建批次
            loader_for_collate = DataLoader(
                batch_data_list, batch_size=current_batch_size)
            batch_for_model = next(iter(loader_for_collate)).to(torch_device)

            pred_molecule_list, true_molecule_list, products_list, scores, nlls, ells = model.sample_batch(
                data=batch_for_model, batch_id=mol_idx, batch_size=current_batch_size, save_final=0, keep_chain=0,
                number_chain_steps_to_save=1, sample_idx=batch_num, save_true_reactants=True, use_one_hot=False
            )

            for i in range(current_batch_size):
                all_samples_for_this_molecule.append({
                    "pred": pred_molecule_list[i], "true": true_molecule_list[i], "prod": products_list[i],
                    "score": scores[i], "nll": nlls[i], "ell": ells[i]
                })

        # --- 8. 处理并写入当前分子的所有结果 ---
        if not all_samples_for_this_molecule:
            continue

        results_to_write = []
        # 获取一次 product smi
        product_smi_for_log = Chem.MolToSmiles(build_molecule(
            all_samples_for_this_molecule[0]["prod"][0], all_samples_for_this_molecule[0]["prod"][1], dataset_infos.atom_decoder))

        for sample in all_samples_for_this_molecule:
            true_mol_built, true_n_dummy_atoms = build_molecule(
                sample["true"][0], sample["true"][1], dataset_infos.atom_decoder, return_n_dummy_atoms=True)
            true_smi = Chem.MolToSmiles(true_mol_built)
            product_mol_built = build_molecule(
                sample["prod"][0], sample["prod"][1], dataset_infos.atom_decoder)
            product_smi = Chem.MolToSmiles(product_mol_built)
            pred_mol_built, n_dummy_atoms = build_molecule(
                sample["pred"][0], sample["pred"][1], dataset_infos.atom_decoder, return_n_dummy_atoms=True)
            pred_smi = Chem.MolToSmiles(pred_mol_built)

            results_to_write.append({
                'product': product_smi, 'pred': pred_smi, 'true': true_smi, 'score': sample["score"],
                'true_n_dummy_nodes': RetroBridgeDatasetInfos.max_n_dummy_nodes - true_n_dummy_atoms,
                'sampled_n_dummy_nodes': RetroBridgeDatasetInfos.max_n_dummy_nodes - n_dummy_atoms,
                'nll': sample["nll"], 'ell': sample["ell"]
            })

        molecule_table = pd.DataFrame(results_to_write)
        molecule_table.to_csv(table_path, mode='a', header=False, index=False)
        tqdm.write(
            f"分子 {mol_idx + 1}/{len(sliced_dataset)} (SMILES: {product_smi_for_log}) 的 {n_samples_per_input} 条采样结果已写入。")

    print(f"所有采样任务完成。")


if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser(description="按分子写入，并行采样版")

    parser.add_argument('--gpu-id', type=int, required=True,
                        help='要使用的GPU的索引 (例如, 1 代表第二张卡)')
    parser.add_argument('--start', type=int, required=True,
                        help='数据切片的起始索引 (包含)')
    parser.add_argument('--stop', type=int, required=True,
                        help='数据切片的结束索引 (不包含)')
    parser.add_argument('--n-samples', type=int,
                        default=100, help='每个输入分子的总采样次数')
    parser.add_argument('--sample-batch-size', type=int,
                        default=64, help='一次并行采样的数量')

    args = parser.parse_args()

    checkpoint_path = 'checkpoints/retrobridge_30_06_08_18_58/top_5_accuracy/epoch=309_top_5_accuracy=0.883.ckpt'
    data_slice = slice(args.start, args.stop)

    main(
        gpu_id=args.gpu_id,
        data_slice=data_slice,
        n_samples_per_input=args.n_samples,
        sample_batch_size=args.sample_batch_size,
        checkpoint_path=checkpoint_path
    )
