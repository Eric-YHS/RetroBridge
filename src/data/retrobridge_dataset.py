import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import subprocess
from pathlib import Path 

from rdkit import Chem
from src.data import utils # Assuming utils.py is in the same directory or accessible
from src.data.abstract_dataset import MolecularDataModule
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import Any, Sequence

from pdb import set_trace

DOWNLOAD_URL_TEMPLATE = 'https://zenodo.org/record/8114657/files/{fname}?download=1'
USPTO_MIT_DOWNLOAD_URL = 'https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip'


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class RetroBridgeDataset(InMemoryDataset):
    types = {
        'N': 0, 'C': 1, 'O': 2, 'S': 3, 'Cl': 4, 'F': 5, 'B': 6, 'Br': 7, 'P': 8,
        'Si': 9, 'I': 10, 'Sn': 11, 'Mg': 12, 'Cu': 13, 'Zn': 14, 'Se': 15, '*': 16,
    }

    bonds = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3
    }

    def __init__(self, stage, root, extra_nodes=False, swap=False, class_to_idx=None, num_reaction_classes=None): 
        self.stage = stage
        self.extra_nodes = extra_nodes
        self.class_to_idx = class_to_idx
        self.num_reaction_classes = num_reaction_classes
        if self.stage == 'train' and (self.class_to_idx is None or self.num_reaction_classes is None):
            pass

        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        elif self.stage == 'test':
            self.file_idx = 2
        else:
            raise NotImplementedError

        # 1. 调用父类构造函数。
        #    它会设置好所有路径属性(如 self.root, self.processed_dir, self.processed_paths)。
        #    我们传递 transform=None 等参数，以确保它遵循标准初始化路径。
        super().__init__(root=root, transform=None, pre_transform=None, pre_filter=None)

        # 2. 我们现在自己控制处理和加载的流程，以解决 torch.load 的版本问题。
        #    检查所有必需的已处理文件是否存在。如果任何一个不存在，就重新处理所有数据。
        #    self.processed_paths 是由 super().__init__ 设置好的属性。
        #    (注意: PyG 的 `InMemoryDataset` 通常只期望一个处理文件，但这里的逻辑是为多个文件设计的)
        #    为了简化，我们只检查当前 stage 需要的文件。如果它不存在，我们就认为需要重新处理。
        if not os.path.exists(self.processed_paths[self.file_idx]):
            print(f"Processed data for stage '{self.stage}' not found. Reprocessing all splits...")
            # process() 方法应该为所有 splits (train, val, test) 创建文件。
            # 这里我们只调用一次 process()，它内部应该处理所有情况。
            # 根据你的代码，process() 只处理 self.file_idx 对应的文件，这是非标准的。
            # 但为了保持你代码的原有逻辑，我们只在需要时调用它。
            # 这是一个设计上的权衡。
            self.process() # 这将只处理当前 stage 的数据，这是你原有代码的逻辑。

        # 3. 现在，我们用我们自己的 torch.load 来加载正确的文件，并附带 `weights_only=False`
        #    这会绕过 PyG 内部加载失败的问题。
        try:
            self.data, self.slices = torch.load(self.processed_paths[self.file_idx], weights_only=False)
        except FileNotFoundError:
            raise RuntimeError(f"File not found: {self.processed_paths[self.file_idx]}. "
                               "Processing may have failed or did not create the expected files.")
        except Exception as e:
            # 捕获其他可能的加载错误
            print(f"An error occurred while loading processed data: {e}")
            raise e

        # 4. swap 逻辑保持不变
        if swap:
            new_data_args = {
                'x': self.data.p_x, 'edge_index': self.data.p_edge_index, 'edge_attr': self.data.p_edge_attr,
                'p_x': self.data.x, 'p_edge_index': self.data.edge_index, 'p_edge_attr': self.data.edge_attr,
                'y': self.data.y, 'idx': self.data.idx, 'r_smiles': self.data.p_smiles, 'p_smiles': self.data.r_smiles,
            }
            if hasattr(self.data, 'reaction_class'):
                new_data_args['reaction_class'] = self.data.reaction_class
            
            self.data = Data(**new_data_args)
            
            new_slices_args = { # Rebuild slices carefully
                'x': self.slices['p_x'],
                'edge_index': self.slices['p_edge_index'],
                'edge_attr': self.slices['p_edge_attr'],
                'y': self.slices['y'],
                'idx': self.slices['idx'],
                'p_x': self.slices['x'],
                'p_edge_index': self.slices['edge_index'],
                'p_edge_attr': self.slices['edge_attr'],
                'r_smiles': self.slices['p_smiles'], # Key was p_smiles, value was r_smiles
                'p_smiles': self.slices['r_smiles'], # Key was r_smiles, value was p_smiles
            }
            if hasattr(self.data, 'reaction_class') and 'reaction_class' in self.slices:
                 new_slices_args['reaction_class'] = self.slices['reaction_class'] # Keep original if present
            self.slices = new_slices_args


    @property
    def processed_dir(self) -> str:
        # ---- MODIFICATION START: Adding a unique identifier for new processing logic ----
        # If you want to ensure re-processing due to class label addition,
        # you can change the processed_dir name slightly.
        # For example, add a suffix like "_v2" or "_with_class".
        # This forces re-processing if the old directory exists without this suffix.
        # However, for "no deletion" principle, we might rely on deleting old files manually.
        # For now, let's keep it as is, assuming users delete old `processed_retrobridge*` dirs.
        # ---- MODIFICATION END ----
        if self.extra_nodes:
            return os.path.join(self.root, f'processed_retrobridge_extra_nodes')
        else:
            return os.path.join(self.root, f'processed_retrobridge')

    @property
    def raw_file_names(self):
        return ['uspto50k_train.csv', 'uspto50k_val.csv', 'uspto50k_test.csv']

    @property
    def split_file_name(self): # This seems redundant with raw_file_names for USPTO-50k
        return ['uspto50k_train.csv', 'uspto50k_val.csv', 'uspto50k_test.csv']

    @property
    def split_paths(self):
        files = to_list(self.split_file_name)
        return [os.path.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return [f'train.pt', f'val.pt', f'test.pt']

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        for fname in self.raw_file_names:
            print(f'Downloading {fname}')
            url = DOWNLOAD_URL_TEMPLATE.format(fname=fname)
            path = os.path.join(self.raw_dir, fname)
            subprocess.run(f'wget {url} -O {path}', shell=True, check=False) # Added check=False

    def process(self): # This method is called by InMemoryDataset if processed_paths don't exist
        print(f"Processing raw data for stage: {self.stage}, file_idx: {self.file_idx}") # Added print
        table = pd.read_csv(self.split_paths[self.file_idx])
        reaction_classes_from_csv = table['class'].values # Assuming 'class' column exists
        
        data_list = []
        skipped_count = 0 # Counter for skipped items
        for i, reaction_smiles in enumerate(tqdm(table['reactants>reagents>production'].values, desc=f"Processing {self.stage} data")):
            original_class_label = reaction_classes_from_csv[i]
            class_idx_tensor = torch.tensor([-1], dtype=torch.long) 

            # num_reaction_classes should be known here if Dataset is called from DataModule after DatasetInfos init.
            # class_to_idx might also be available.
            # If self.num_reaction_classes is None, it means this Dataset instance wasn't properly
            # initialized with this info (e.g. direct instantiation outside DataModule).
            if self.num_reaction_classes is not None and self.num_reaction_classes > 0 :
                try:
                    class_val = int(original_class_label) # USPTO-50k classes are 1-10
                    mapped_idx = class_val - 1 # Convert to 0-indexed
                    if 0 <= mapped_idx < self.num_reaction_classes:
                        class_idx_tensor = torch.tensor([mapped_idx], dtype=torch.long)
                    # else:
                        # print(f"Debug: Class label {original_class_label} (mapped to {mapped_idx}) out of range [0, {self.num_reaction_classes-1}]. Using -1 for sample {i}, stage {self.stage}.")
                except ValueError:
                    # print(f"Debug: Could not convert class label '{original_class_label}' to int for sample {i}, stage {self.stage}. Using -1.")
                    pass 
            # else:
                # print(f"Debug: num_reaction_classes is None or invalid ({self.num_reaction_classes}) during process for stage {self.stage}. Class index will be -1 for sample {i}.")


            reactants_smi, _, product_smi = reaction_smiles.split('>')
            rmol = Chem.MolFromSmiles(reactants_smi)
            pmol = Chem.MolFromSmiles(product_smi)

            if rmol is None or pmol is None: 
                skipped_count += 1
                continue

            r_num_nodes_orig = rmol.GetNumAtoms() # Renamed to avoid conflict
            p_num_nodes = pmol.GetNumAtoms()
            
            if not (p_num_nodes <= r_num_nodes_orig): # Changed from assert to if-continue
                # print(f"Warning (process): Product has more atoms ({p_num_nodes}) than reactant ({r_num_nodes_orig}) for reaction {i}. Skipping.")
                skipped_count += 1
                continue


            r_num_nodes_for_graph = r_num_nodes_orig # Default

            if self.extra_nodes:
                new_r_num_nodes = p_num_nodes + RetroBridgeDatasetInfos.max_n_dummy_nodes # Static access
                if r_num_nodes_orig > new_r_num_nodes:
                    if self.stage in ['train', 'val']:
                        skipped_count += 1
                        continue
                    else: # Test stage, use placeholder 'C'
                        reactants_smi_orig, product_smi_orig = reactants_smi, product_smi # Save originals for p_smiles, r_smiles
                        reactants_smi, product_smi = 'C', 'C'
                        rmol = Chem.MolFromSmiles(reactants_smi)
                        pmol = Chem.MolFromSmiles(product_smi)
                        if rmol is None or pmol is None: # Should not happen with 'C'
                             skipped_count +=1
                             continue
                        p_num_nodes = pmol.GetNumAtoms()
                        r_num_nodes_orig = rmol.GetNumAtoms() # Update r_num_nodes_orig for 'C'
                        new_r_num_nodes = p_num_nodes + RetroBridgeDatasetInfos.max_n_dummy_nodes
                        product_smi = product_smi_orig # Keep original product SMILES for data object
                        reactants_smi = reactants_smi_orig # Keep original reactant SMILES for data object

                r_num_nodes_for_graph = new_r_num_nodes
            else:
                r_num_nodes_for_graph = r_num_nodes_orig


            try:
                mapping_r = self.compute_nodes_order_mapping(rmol) # For reactants
                r_x, r_edge_index, r_edge_attr = self.compute_graph(
                    rmol, mapping_r, r_num_nodes_for_graph, types=self.types, bonds=self.bonds
                )
                # For products, the number of nodes in its graph should be consistent with r_num_nodes_for_graph
                # if they are to be compared or used in a bridge/diffusion.
                # The mapping should be from the reactant's perspective if atoms are conserved.
                # However, product molecule (pmol) might have different atom indices than rmol,
                # but atom_map_numbers should be consistent for mapped atoms.
                # The original compute_graph uses mapping for node indexing.
                # We need a mapping that is valid for pmol as well for conserved atoms.
                # Let's assume mapping_r (based on rmol's map numbers) is used for both,
                # and compute_graph correctly handles missing atoms in pmol by assigning dummy type.
                mapping_p = self.compute_nodes_order_mapping(pmol) # Mapping for product based on its own atoms initially
                                                                 # This needs careful thought. If mapping_r is the canonical order,
                                                                 # pmol should be expressed in that order.
                                                                 # Let's assume mapping_r is the global order derived from reactants.
                
                # Re-evaluate how mapping is used:
                # The mapping from compute_nodes_order_mapping creates a dense 0..N-1 indexing
                # based on sorted atom_map_numbers PRESENT in the molecule.
                # If rmol and pmol share atom_map_numbers, these numbers should map to the same
                # index in the *final dense graph representation* if we want consistency.
                # The current approach creates separate mappings if called on rmol and pmol independently.
                # For RetroBridge, the graph size is often determined by r_num_nodes_for_graph (max of reactant or product+dummy).
                # Let's use mapping_r as the reference mapping for atom map numbers.
                
                p_x, p_edge_index, p_edge_attr = self.compute_graph(
                    pmol, mapping_r, r_num_nodes_for_graph, types=self.types, bonds=self.bonds
                )

            except KeyError as e: # Catch specific KeyError from mapping
                # print(f"KeyError during graph computation for reaction {i}: {e}. Atom map number likely missing or inconsistent. Skipping.")
                skipped_count += 1
                continue
            except Exception as e:
                # print(f'Error processing molecule {i} (idx) for graph computation: {e}. Skipping.')
                skipped_count += 1
                continue

            if self.stage in ['train', 'val']: # Original assert
                if not (len(p_x) == len(r_x)):
                    # print(f"Length mismatch for reaction {i}: p_x={len(p_x)}, r_x={len(r_x)}. Skipping.")
                    skipped_count +=1
                    continue


            product_mask_check = ~(p_x[:, -1].bool()).squeeze() # atoms in product that are NOT dummy
            # This check ensures that non-dummy atoms in product have same features as in reactant
            # (implies correct mapping and conservation)
            if len(r_x) == len(p_x) and product_mask_check.any(): # Only check if there are non-dummy product atoms
                if not torch.allclose(r_x[product_mask_check], p_x[product_mask_check]):
                    # print(f'Atom feature mismatch for mapped atoms in reaction {i}. Skipping.')
                    skipped_count += 1
                    continue
            
            # Original: if self.stage == 'train' and len(p_edge_attr) == 0: continue
            # This might skip valid single-atom products. Let's be more specific.
            # Skip if product has no atoms (after filtering dummies) and no edges.
            # num_product_real_atoms = product_mask_check.sum().item()
            # if self.stage == 'train' and num_product_real_atoms > 0 and p_edge_index.numel() == 0 and num_product_real_atoms > 1:
                # This means a multi-atom product with no bonds, which might be an error.
                # print(f"Multi-atom product with no bonds for reaction {i} in train set. Skipping.")
                # skipped_count += 1
                # continue
            # The original `len(p_edge_attr) == 0` check is simpler. Keep it for now.
            if self.stage == 'train' and p_edge_attr.numel() == 0 and p_x[product_mask_check].size(0) > 1: # If product is multi-atom but no edges
                # print(f"Training sample {i} has a multi-atom product with no edges. Skipping.")
                skipped_count += 1
                continue


            # Shuffle nodes to avoid leaking (if X and P have same num nodes for shuffling)
            if len(p_x) == len(r_x) and r_num_nodes_for_graph > 0 : # Add check for >0 nodes
                new2old_idx = torch.randperm(r_num_nodes_for_graph).long()
                old2new_idx = torch.empty_like(new2old_idx)
                old2new_idx[new2old_idx] = torch.arange(r_num_nodes_for_graph)

                r_x = r_x[new2old_idx]
                if r_edge_index.numel() > 0:
                    r_edge_index = torch.stack([old2new_idx[r_edge_index[0]], old2new_idx[r_edge_index[1]]], dim=0)
                r_edge_index, r_edge_attr = self.sort_edges(r_edge_index, r_edge_attr, r_num_nodes_for_graph)

                p_x = p_x[new2old_idx]
                if p_edge_index.numel() > 0:
                    p_edge_index = torch.stack([old2new_idx[p_edge_index[0]], old2new_idx[p_edge_index[1]]], dim=0)
                p_edge_index, p_edge_attr = self.sort_edges(p_edge_index, p_edge_attr, r_num_nodes_for_graph)

                # Re-check after shuffling
                product_mask_shuffled = ~(p_x[:, -1].bool()).squeeze()
                if product_mask_shuffled.any(): # if any non-dummy atoms
                    if not torch.allclose(r_x[product_mask_shuffled], p_x[product_mask_shuffled]):
                        # This should ideally not happen if logic is correct.
                        # print(f"Atom feature mismatch after shuffling for reaction {i}. Skipping.")
                        skipped_count += 1
                        continue

            y_data = torch.zeros(size=(1, 0), dtype=torch.float) # Renamed variable
            data_obj = Data( # Renamed variable
                x=r_x, edge_index=r_edge_index, edge_attr=r_edge_attr, y=y_data, idx=i,
                p_x=p_x, p_edge_index=p_edge_index, p_edge_attr=p_edge_attr,
                r_smiles=reactants_smi, p_smiles=product_smi, # Ensure these are original if 'C' was substituted
                reaction_class=class_idx_tensor
            )
            data_list.append(data_obj)

        print(f'Dataset {self.stage} contains {len(data_list)} reactions after processing (skipped {skipped_count}).')
        if not data_list: 
            print(f"Critical Warning: data_list for stage {self.stage} is empty. This WILL cause issues.")
            # Create a dummy data item to prevent collate from failing, though this hides the problem.
            # A better approach is to ensure the processing pipeline has valid fallbacks or error handling
            # that doesn't lead to an entirely empty list if some data is expected.
            # For now, this ensures collate doesn't fail immediately on an empty list.
            # However, an empty dataset is a major issue.
            dummy_x = F.one_hot(torch.tensor([self.types['C']]), num_classes=len(self.types)).float()
            dummy_data = Data(x=dummy_x, edge_index=torch.empty((2,0), dtype=torch.long), edge_attr=torch.empty((0, len(self.bonds)+1)),
                                p_x=dummy_x.clone(), p_edge_index=torch.empty((2,0), dtype=torch.long), p_edge_attr=torch.empty((0, len(self.bonds)+1)),
                                y=torch.zeros(size=(1,0), dtype=torch.float), idx=0,
                                r_smiles="C", p_smiles="C", reaction_class=torch.tensor([-1], dtype=torch.long)
                               )
            data_list.append(dummy_data)
            print("Added a dummy data item to prevent crash on empty dataset.")

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

    @staticmethod
    def compute_graph(molecule, mapping, max_num_nodes, types, bonds):
        # molecule: RDKit Mol object
        # mapping: dict from atom_map_num to 0..N-1 index
        # max_num_nodes: target size of the graph (for padding with dummy atoms)
        
        # Initialize type_idx with dummy type. Actual size will be max_num_nodes.
        # The dummy type is the last one in `types`.
        dummy_atom_type_idx = len(types) - 1 
        type_idx = [dummy_atom_type_idx] * max_num_nodes
        
        processed_atoms_in_mapping = 0
        for atom in molecule.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom_map_num = atom.GetAtomMapNum()
                if atom_map_num in mapping: # Only consider atoms whose map numbers are in the provided mapping
                    graph_idx = mapping[atom_map_num]
                    if 0 <= graph_idx < max_num_nodes: # Ensure index is within bounds
                        type_idx[graph_idx] = types.get(atom.GetSymbol(), dummy_atom_type_idx) # Use .get for safety
                        processed_atoms_in_mapping += 1
            # else:
                # print(f"Atom {atom.GetIdx()} symbol {atom.GetSymbol()} has no molAtomMapNumber.")
        
        # If no atoms were mapped (e.g. product molecule is empty or has no mapped atoms from reactant)
        # type_idx will remain all dummies.
        # print(f"Debug compute_graph: Processed {processed_atoms_in_mapping} atoms based on mapping for molecule {Chem.MolToSmiles(molecule) if molecule else 'None'}")


        num_classes_atoms = len(types)
        x = F.one_hot(torch.tensor(type_idx), num_classes=num_classes_atoms).float()

        row, col, edge_type = [], [], []
        num_bonds_processed = 0
        for bond in molecule.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            
            if begin_atom.HasProp('molAtomMapNumber') and end_atom.HasProp('molAtomMapNumber'):
                start_atom_map_num = begin_atom.GetAtomMapNum()
                end_atom_map_num = end_atom.GetAtomMapNum()

                if start_atom_map_num in mapping and end_atom_map_num in mapping:
                    start_node_idx = mapping[start_atom_map_num]
                    end_node_idx = mapping[end_atom_map_num]
                    
                    # Ensure indices are within graph bounds
                    if 0 <= start_node_idx < max_num_nodes and 0 <= end_node_idx < max_num_nodes:
                        bond_rdkit_type = bond.GetBondType()
                        if bond_rdkit_type in bonds:
                            edge_type_val = bonds[bond_rdkit_type] + 1 # +1 for 0=no_bond
                            row += [start_node_idx, end_node_idx]
                            col += [end_node_idx, start_node_idx]
                            edge_type += 2 * [edge_type_val]
                            num_bonds_processed +=1
        
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long) # Renamed
        num_classes_bonds = len(bonds) + 1 # +1 for no_bond class
        edge_attr = F.one_hot(edge_type_tensor, num_classes=num_classes_bonds).to(torch.float)

        return x, edge_index, edge_attr

    @staticmethod
    def compute_nodes_order_mapping(molecule):
        order = []
        for atom in molecule.GetAtoms():
            if atom.HasProp('molAtomMapNumber'): # Ensure atom has map number
                order.append(atom.GetAtomMapNum())
            # else:
                # This situation means an atom in the molecule (e.g. reactant) doesn't have a map number.
                # This could be problematic if it's expected to be part of the mapping.
                # print(f"Warning (compute_nodes_order_mapping): Atom {atom.GetIdx()} has no map number.")
        
        if not order: # If no atoms had map numbers
            # print("Warning (compute_nodes_order_mapping): No atom map numbers found in molecule. Mapping will be empty.")
            return {}
            
        order_sorted = sorted(list(set(order))) 
        order_map = {
            atom_map_num: idx
            for idx, atom_map_num in enumerate(order_sorted)
        }
        return order_map

    @staticmethod
    def sort_edges(edge_index, edge_attr, max_num_nodes):
        if edge_index.numel() > 0 and edge_attr.numel() > 0 : # Check both have elements
            if edge_index.shape[1] == edge_attr.shape[0]: # Ensure consistency
                perm = (edge_index[0] * max_num_nodes + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]
            # else:
                # print(f"Warning (sort_edges): Mismatch between edge_index columns ({edge_index.shape[1]}) and edge_attr rows ({edge_attr.shape[0]}). Skipping sort.")
        return edge_index, edge_attr


class RetroBridgeMITDataset(RetroBridgeDataset): # Unchanged as per request
    types = { # This is specific to MIT, different from parent's types.
        'C': 0, 'O': 1, 'N': 2, 'Cl': 3, 'F': 4, 'S': 5, 'Na': 6, 'Br': 7, 'K': 8, 'P': 9, 'H': 10,
        'I': 11, 'B': 12, 'Li': 13, 'Si': 14, 'Pd': 15, 'Cs': 16, 'Al': 17, 'Cu': 18, 'Mg': 19, 'Sn': 20,
        'Zn': 21, 'Fe': 22, 'Cr': 23, 'Mn': 24, 'Ti': 25, 'Pt': 26, 'Ca': 27, 'Ag': 28, 'Se': 29, 'Ni': 30,
        'Ru': 31, 'Rh': 32, 'Co': 33, 'Os': 34, 'Ce': 35, 'Pb': 36, 'Ba': 37, 'Hg': 38, 'Zr': 39, 'As': 40,
        'Yb': 41, 'W': 42, 'Bi': 43, 'Ge': 44, 'In': 45, 'Sb': 46, 'Sc': 47, 'Tl': 48, 'Mo': 49, 'Sm': 50,
        'Re': 51, 'Ir': 52, 'Au': 53, 'Cd': 54, 'Ga': 55, 'Xe': 56, 'Nd': 57, 'Ta': 58, 'V': 59, 'La': 60,
        'Rb': 61, 'Dy': 62, 'Hf': 63, 'Y': 64, 'Te': 65, 'Ar': 66, 'Pr': 67, 'He': 68, 'Be': 69, 'Eu': 70,
        'Sr': 71, '*': 72,
    }
    # Bonds are likely the same, inherited from parent.

    def __init__(self, stage, root, extra_nodes=False, swap=False, class_to_idx=None, num_reaction_classes=None):
        # For MIT, class_to_idx and num_reaction_classes are not used.
        # Pass them as None to the super constructor for the MIT case.
        super().__init__(stage, root, extra_nodes, swap, class_to_idx=None, num_reaction_classes=None)

    @property
    def raw_file_names(self): # Overrides parent
        return ['train.csv', 'valid.csv', 'test.csv'] # These are for MIT

    @property
    def split_file_name(self): # Overrides parent
        return ['train.csv', 'valid.csv', 'test.csv'] # For MIT

    # process method will be inherited, but it uses self.split_paths which now points to MIT files.
    # The `table['class']` access in `process` will fail for MIT data as it doesn't have this column.
    # This needs to be handled if RetroBridgeMITDataset.process is ever called directly
    # and not through a specialized MIT processing logic.
    # However, `RetroBridgeMITDataModule` calls `download` which creates these CSVs from .txt,
    # and those .txt files don't have a 'class' column. The `convert_txt_to_df` also doesn't add it.
    # So, `table['class']` will indeed cause a KeyError for MIT if parent's `process` is used.
    # This means `RetroBridgeMITDataset` MUST override `process` or the parent `process` needs to be more generic.
    # Given "uspto-mit不用管", we assume this discrepancy is handled or MIT is not used with this exact flow.
    # For robustness, a simple override for `process` in MIT can be added if it's an issue.
    def process(self):
        # Specialized processing for MIT data if needed, or call super().process()
        # with a flag or try-except for the 'class' column if parent's process is to be reused.
        # For now, keeping it as inheriting, which means it WILL TRY to use parent's process logic.
        # This is a known point of failure if used directly for MIT without addressing the 'class' column.
        # A quick fix to prevent crash for MIT, assuming no class labels for MIT:
        print(f"Processing MIT data for stage: {self.stage}, file_idx: {self.file_idx}")
        table = pd.read_csv(self.split_paths[self.file_idx]) # MIT files are train.csv, valid.csv, test.csv from USPTO_MIT_DOWNLOAD_URL
        
        # MIT data does not have a 'class' column in the same way.
        # We will effectively assign a dummy class or handle its absence.
        # The `self.num_reaction_classes` for MIT would be None or 0 from its DatasetInfos.
        
        data_list = []
        skipped_count = 0
        for i, reaction_smiles in enumerate(tqdm(table['reactants>reagents>production'].values, desc=f"Processing MIT {self.stage} data")):
            # For MIT, reaction_class is not applicable.
            class_idx_tensor = torch.tensor([-1], dtype=torch.long) # Default for no class info

            reactants_smi, _, product_smi = reaction_smiles.split('>')
            # Mol sanitization and atom mapping are critical for MIT data as well.
            # Re-using parent's compute_graph, compute_nodes_order_mapping logic.
            # Atom types for MIT are different (self.types in this class).
            # Bond types are same.

            rmol = Chem.MolFromSmiles(reactants_smi)
            pmol = Chem.MolFromSmiles(product_smi)

            if rmol is None or pmol is None:
                skipped_count += 1
                continue

            r_num_nodes_orig = rmol.GetNumAtoms()
            p_num_nodes = pmol.GetNumAtoms()
            
            if not (p_num_nodes <= r_num_nodes_orig):
                skipped_count += 1
                continue

            r_num_nodes_for_graph = r_num_nodes_orig
            if self.extra_nodes: # Using parent's RetroBridgeDatasetInfos for max_n_dummy_nodes for now
                new_r_num_nodes = p_num_nodes + RetroBridgeDatasetInfos.max_n_dummy_nodes 
                if r_num_nodes_orig > new_r_num_nodes:
                    if self.stage in ['train', 'val']:
                        skipped_count += 1
                        continue
                    else:
                        reactants_smi_orig, product_smi_orig = reactants_smi, product_smi
                        reactants_smi, product_smi = 'CC', 'CC' # MIT uses 'CC' not 'C' in some contexts
                        rmol = Chem.MolFromSmiles(reactants_smi)
                        pmol = Chem.MolFromSmiles(product_smi)
                        if rmol is None or pmol is None: skipped_count +=1; continue
                        p_num_nodes = pmol.GetNumAtoms()
                        r_num_nodes_orig = rmol.GetNumAtoms()
                        new_r_num_nodes = p_num_nodes + RetroBridgeDatasetInfos.max_n_dummy_nodes
                        product_smi = product_smi_orig 
                        reactants_smi = reactants_smi_orig
                r_num_nodes_for_graph = new_r_num_nodes
            else:
                r_num_nodes_for_graph = r_num_nodes_orig
            
            try:
                # IMPORTANT: Must use self.types (MIT specific) and self.bonds (parent's)
                mapping_r = self.compute_nodes_order_mapping(rmol)
                r_x, r_edge_index, r_edge_attr = self.compute_graph(
                    rmol, mapping_r, r_num_nodes_for_graph, types=self.types, bonds=self.bonds
                )
                p_x, p_edge_index, p_edge_attr = self.compute_graph(
                    pmol, mapping_r, r_num_nodes_for_graph, types=self.types, bonds=self.bonds
                )
            except Exception as e:
                skipped_count += 1
                continue

            # Validations similar to parent's process
            if self.stage in ['train', 'val']:
                if not (len(p_x) == len(r_x)): skipped_count +=1; continue
            
            product_mask_check = ~(p_x[:, -1].bool()).squeeze()
            if len(r_x) == len(p_x) and product_mask_check.any():
                if not torch.allclose(r_x[product_mask_check], p_x[product_mask_check]):
                    skipped_count += 1; continue
            
            if self.stage == 'train' and p_edge_attr.numel() == 0 and p_x[product_mask_check].size(0) > 1:
                skipped_count += 1; continue

            if len(p_x) == len(r_x) and r_num_nodes_for_graph > 0:
                new2old_idx = torch.randperm(r_num_nodes_for_graph).long()
                old2new_idx = torch.empty_like(new2old_idx)
                old2new_idx[new2old_idx] = torch.arange(r_num_nodes_for_graph)

                r_x = r_x[new2old_idx]
                if r_edge_index.numel() > 0: r_edge_index = torch.stack([old2new_idx[r_edge_index[0]], old2new_idx[r_edge_index[1]]], dim=0)
                r_edge_index, r_edge_attr = self.sort_edges(r_edge_index, r_edge_attr, r_num_nodes_for_graph)

                p_x = p_x[new2old_idx]
                if p_edge_index.numel() > 0: p_edge_index = torch.stack([old2new_idx[p_edge_index[0]], old2new_idx[p_edge_index[1]]], dim=0)
                p_edge_index, p_edge_attr = self.sort_edges(p_edge_index, p_edge_attr, r_num_nodes_for_graph)
                
                product_mask_shuffled = ~(p_x[:, -1].bool()).squeeze()
                if product_mask_shuffled.any():
                    if not torch.allclose(r_x[product_mask_shuffled], p_x[product_mask_shuffled]):
                        skipped_count += 1; continue
            
            y_data = torch.zeros(size=(1, 0), dtype=torch.float)
            data_obj = Data(
                x=r_x, edge_index=r_edge_index, edge_attr=r_edge_attr, y=y_data, idx=i,
                p_x=p_x, p_edge_index=p_edge_index, p_edge_attr=p_edge_attr,
                r_smiles=reactants_smi, p_smiles=product_smi,
                reaction_class=class_idx_tensor # Will be -1 for MIT
            )
            data_list.append(data_obj)

        print(f'Dataset MIT {self.stage} contains {len(data_list)} reactions after processing (skipped {skipped_count}).')
        if not data_list:
            print(f"Critical Warning: MIT data_list for stage {self.stage} is empty.")
            # Add dummy data if necessary, similar to parent's process method
            dummy_x_mit = F.one_hot(torch.tensor([self.types['C']]), num_classes=len(self.types)).float() # Use MIT 'C'
            dummy_data_mit = Data(x=dummy_x_mit, edge_index=torch.empty((2,0), dtype=torch.long), edge_attr=torch.empty((0, len(self.bonds)+1)),
                                p_x=dummy_x_mit.clone(), p_edge_index=torch.empty((2,0), dtype=torch.long), p_edge_attr=torch.empty((0, len(self.bonds)+1)),
                                y=torch.zeros(size=(1,0), dtype=torch.float), idx=0,
                                r_smiles="CC", p_smiles="CC", reaction_class=torch.tensor([-1], dtype=torch.long)
                               )
            data_list.append(dummy_data_mit)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])


    @staticmethod
    def convert_txt_to_df(path): # Unchanged
        reactions = []
        with open(path, 'r') as f:
            for line in tqdm(list(f.readlines()), desc="Converting MIT txt to df"):
                rxn_parts = line.strip().split() # Split by space
                if rxn_parts: # Ensure line is not empty
                    rxn = rxn_parts[0] # Take the first part which is the reaction SMILES
                    reactions.append(rxn)
        return pd.DataFrame({
            'reactants>reagents>production': reactions
        })

    def download(self): # Unchanged
        os.makedirs(self.raw_dir, exist_ok=True)
        path = os.path.join(self.raw_dir, 'data.zip')
        print(f"Downloading MIT data to {path}...") # Added print
        subprocess.run(f'wget {USPTO_MIT_DOWNLOAD_URL} -O {path}', shell=True, check=False)
        print(f"Unzipping MIT data in {self.raw_dir}...") # Added print
        subprocess.run(f'unzip -o {path} -d {self.raw_dir}', shell=True, check=False) # Added -o to overwrite

        for name in ['train', 'test', 'valid']:
            src_path = os.path.join(self.raw_dir, 'data', f'{name}.txt')
            dst_path = os.path.join(self.raw_dir, f'{name}.csv')
            if os.path.exists(src_path): # Check if source txt exists
                table = self.convert_txt_to_df(src_path)
                table.to_csv(dst_path, index=False)
            # else:
                # print(f"Warning (MIT download): Source file {src_path} not found after unzipping.")


class RetroBridgeDataModule(MolecularDataModule):
    DATASET_CLASS = RetroBridgeDataset

    def __init__(self, data_root, batch_size, num_workers, shuffle, extra_nodes=False, evaluation=False, swap=False):
        super().__init__(batch_size, num_workers, shuffle) 
        self.extra_nodes = extra_nodes
        self.evaluation = evaluation 
        self.swap = swap
        self.data_root = data_root
        self.train_smiles = []
        
        # ---- MODIFICATION FOR STAGED INITIALIZATION ----
        # STAGE 1: Initialize DatasetInfos with minimal info (not needing dataloaders yet)
        # Pass only what's needed for _init_early_attributes
        self.dataset_infos = RetroBridgeDatasetInfos(
            datamodule_data_root=self.data_root,
            datamodule_extra_nodes=self.extra_nodes,
            datamodule_is_mit=isinstance(self, RetroBridgeMITDataModule) # Check if it's MIT variant
        )
        # At this point, self.dataset_infos.num_reaction_classes and class_to_idx are populated (for USPTO-50k).

        # STAGE 2: Call prepare_data(). This will instantiate RetroBridgeDataset,
        # which needs num_reaction_classes and class_to_idx from self.dataset_infos.
        # RetroBridgeDataset.process() will then run if processed files are missing.
        self.prepare_data() 
        
        # STAGE 3: Now that dataloaders are ready (created in self.prepare_data),
        # complete the initialization of dataset_infos with stats derived from dataloaders.
        if hasattr(self.dataset_infos, 'complete_init_with_dataloaders'):
            self.dataset_infos.complete_init_with_dataloaders(self)
        else:
            print("Warning: dataset_infos object does not have 'complete_init_with_dataloaders' method.")
        # ---- END OF MODIFICATION ----

    def prepare_data(self) -> None: 
        stage_for_dataset = 'val' if self.evaluation else 'train' 
        
        # These are now available from the _init_early_attributes call
        class_info_idx = self.dataset_infos.class_to_idx 
        class_info_num = self.dataset_infos.num_reaction_classes

        datasets = {
            'train': self.DATASET_CLASS(
                stage=stage_for_dataset, root=self.data_root, extra_nodes=self.extra_nodes, swap=self.swap,
                class_to_idx=class_info_idx,
                num_reaction_classes=class_info_num
            ),
            'val': self.DATASET_CLASS(
                stage='val', root=self.data_root, extra_nodes=self.extra_nodes, swap=self.swap,
                class_to_idx=class_info_idx,
                num_reaction_classes=class_info_num
            ),
            'test': self.DATASET_CLASS(
                stage='test', root=self.data_root, extra_nodes=self.extra_nodes, swap=self.swap,
                class_to_idx=class_info_idx,
                num_reaction_classes=class_info_num
            ),
        }

        self.dataloaders = {}
        for split, dataset_obj in datasets.items(): 
            self.dataloaders[split] = DataLoader(
                dataset=dataset_obj,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=(self.shuffle and split == 'train'),
            )
        
        if datasets['train'].data is not None and hasattr(datasets['train'].data, 'r_smiles'):
            if isinstance(datasets['train'].data.r_smiles, list):
                 self.train_smiles = datasets['train'].data.r_smiles


class RetroBridgeMITDataModule(RetroBridgeDataModule): # Inherits from modified RetroBridgeDataModule
    DATASET_CLASS = RetroBridgeMITDataset
    
    def __init__(self, data_root, batch_size, num_workers, shuffle, extra_nodes=False, evaluation=False, swap=False):
        # Call MolecularDataModule's init directly
        super(RetroBridgeDataModule, self).__init__(batch_size, num_workers, shuffle) 
        
        self.extra_nodes = extra_nodes
        self.evaluation = evaluation 
        self.swap = swap
        self.data_root = data_root
        self.train_smiles = []
        
        # ---- STAGED INITIALIZATION FOR MIT ----
        # STAGE 1: Initialize MITDatasetInfos with minimal info
        self.dataset_infos = RetroBridgeMITDatasetInfos( # Use MIT specific Infos
            datamodule_data_root=self.data_root,
            datamodule_extra_nodes=self.extra_nodes,
            datamodule_is_mit=True # Explicitly True
        )
        # For MIT, _init_early_attributes in MITDatasetInfos does not compute class info from uspto50k_train.csv.

        # STAGE 2: Call prepare_data(). This will use self.DATASET_CLASS = RetroBridgeMITDataset.
        # RetroBridgeMITDataset.process() is overridden and does not rely on class_to_idx from dataset_infos.
        self.prepare_data()
        
        # STAGE 3: Complete MITDatasetInfos initialization with stats from dataloaders.
        if hasattr(self.dataset_infos, 'complete_init_with_dataloaders'):
            self.dataset_infos.complete_init_with_dataloaders(self)
        else:
            print("Warning: MIT dataset_infos object does not have 'complete_init_with_dataloaders' method.")
        # ---- END OF STAGED INITIALIZATION ----

class RetroBridgeDatasetInfos:
    atom_encoder = RetroBridgeDataset.types 
    atom_decoder = [k for k, v in sorted(RetroBridgeDataset.types.items(), key=lambda item: item[1])] 
    max_n_dummy_nodes = 10

    def __init__(self, datamodule_data_root: str, datamodule_extra_nodes: bool, datamodule_is_mit: bool): # Pass specific primitive attributes
        self.name = 'USPTO-MIT-RetroBridge' if datamodule_is_mit else 'USPTO50K-RetroBridge'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = True 
        self.max_weight = 1000
        self.possible_num_dummy_nodes = list(range(RetroBridgeDatasetInfos.max_n_dummy_nodes + 1))

        self.valencies = None 
        self.atom_weights = None 

        self.dummy_nodes_dist = None
        self.n_nodes = None
        self.max_n_nodes = None # Will be computed
        self.node_types = None
        self.edge_types = None
        self.valency_distribution = None
        self.nodes_dist = None
        
        self.num_reaction_classes = None
        self.class_to_idx = None
        self.idx_to_class = None
        
        # STAGE 1: Initialize attributes not dependent on dataloaders
        self._init_early_attributes(datamodule_data_root, datamodule_is_mit)

    def _init_early_attributes(self, data_root: str, is_mit: bool):
        # This method computes attributes that can be derived from raw files or are static.
        if not is_mit: # USPTO-50k specific logic
            self.valencies = [5, 4, 6, 6, 7, 1, 3, 7, 5, 4, 7, 4, 2, 4, 2, 6, 0]
            self.atom_weights = {
                0: 14.01, 1: 12.01, 2: 16.00, 3: 32.07, 4: 35.45, 5: 19.00, 6: 10.81, 7: 79.90, 
                8: 30.97, 9: 28.09, 10: 126.90, 11: 118.71, 12: 24.31, 13: 63.55, 14: 65.38, 
                15: 78.97, 16: 0.0
            }
            try:
                train_df_path = Path(data_root) / 'raw' / 'uspto50k_train.csv'
                if train_df_path.exists():
                    train_df = pd.read_csv(train_df_path)
                    raw_classes = sorted(train_df['class'].astype(int).unique())
                    if raw_classes and raw_classes[0] == 1:
                        self.num_reaction_classes = int(raw_classes[-1])
                        self.class_to_idx = {orig_class: orig_class - 1 for orig_class in raw_classes}
                        self.idx_to_class = {idx: orig_class for orig_class, idx in self.class_to_idx.items()}
                        print(f"USPTO-50k (early_init): Determined {self.num_reaction_classes} classes.")
                    else:
                        self.class_to_idx = {cls_val: i for i, cls_val in enumerate(raw_classes)}
                        self.idx_to_class = {i: cls_val for cls_val, i in self.class_to_idx.items()}
                        self.num_reaction_classes = len(raw_classes)
                        print(f"USPTO-50k (early_init): Determined {self.num_reaction_classes} classes from unique values.")
                else:
                    print(f"Warning (DatasetInfos early_init): Train CSV {train_df_path} not found.")
                    self.num_reaction_classes = 10 # Fallback
                    print(f"Falling back to num_reaction_classes = {self.num_reaction_classes}")
            except Exception as e:
                print(f"Error computing USPTO-50k class info (early_init): {e}")
                self.num_reaction_classes = 10 # Fallback
                print(f"Falling back to num_reaction_classes = {self.num_reaction_classes} due to error.")
        else: # MIT dataset
            self.name = 'USPTO-MIT-RetroBridge'
            # MIT specific static initializations if any (valencies, atom_weights are None for MIT here)

    # STAGE 2: Initialize attributes dependent on dataloaders
    def complete_init_with_dataloaders(self, datamodule: MolecularDataModule): # Changed type hint
        # This method will be called from DataModule *after* dataloaders are ready.
        # Determine info_dir (copied from old init_attributes)
        info_dir_base = getattr(datamodule, 'data_root', f'./data/{self.name.split("-")[0].lower()}')
        extra_nodes_flag = getattr(datamodule, 'extra_nodes', False)
        
        info_dir_suffix = "_extra_nodes" if extra_nodes_flag else ""
        if "MIT" in self.name:
             info_dir = os.path.join(info_dir_base, f'info_retrobridge_mit{info_dir_suffix}')
        else:
             info_dir = os.path.join(info_dir_base, f'info_retrobridge{info_dir_suffix}')
        os.makedirs(info_dir, exist_ok=True)

        required_files_exist = all(
            os.path.exists(os.path.join(info_dir, fname))
            for fname in ['dummy_nodes_dist.txt', 'n_counts.txt', 'atom_types.txt', 'edge_types.txt']
        )
        valencies_file_exists = os.path.exists(os.path.join(info_dir, 'valencies.txt'))

        if getattr(datamodule, 'evaluation', False) and required_files_exist:
            print(f"Info (DatasetInfos for {self.name}): Loading pre-computed stats from {info_dir} (evaluation mode).")
            try:
                self.dummy_nodes_dist = torch.from_numpy(np.loadtxt(os.path.join(info_dir, 'dummy_nodes_dist.txt'))).float()
                self.n_nodes = torch.from_numpy(np.loadtxt(os.path.join(info_dir, 'n_counts.txt'))).float()
                if self.n_nodes.numel() > 0 : self.max_n_nodes = len(self.n_nodes) - 1
                else: self.max_n_nodes = 0
                self.node_types = torch.from_numpy(np.loadtxt(os.path.join(info_dir, 'atom_types.txt'))).float()
                self.edge_types = torch.from_numpy(np.loadtxt(os.path.join(info_dir, 'edge_types.txt'))).float()
                if valencies_file_exists and "MIT" not in self.name : # Only load valencies for non-MIT if file exists
                    self.valency_distribution = torch.from_numpy(np.loadtxt(os.path.join(info_dir, 'valencies.txt'))).float()
                else: self.valency_distribution = None
                if self.n_nodes.numel() > 0 : self.nodes_dist = utils.DistributionNodes(self.n_nodes)
                else: self.nodes_dist = None
            except Exception as e:
                print(f"Error loading pre-computed stats for {self.name} from {info_dir}: {e}. Will re-compute.")
                self._compute_and_save_distributions(datamodule, info_dir)
        else:
            print(f"Info (DatasetInfos for {self.name}): Computing stats (not eval mode or files missing for {self.name}).")
            self._compute_and_save_distributions(datamodule, info_dir)
            
    # _compute_and_save_distributions and compute_input_output_dims remain the same as previous version.
    # ... (keep _compute_and_save_distributions method as it was in the last iteration)
    def _compute_and_save_distributions(self, datamodule: MolecularDataModule, info_dir: str):
        if hasattr(datamodule, 'dataloaders') and datamodule.dataloaders and datamodule.train_dataloader() is not None : # Check train_dataloader
            print(f"Info ({self.name}): Computing distributions using available dataloaders.")
            self.dummy_nodes_dist = datamodule.dummy_atoms_counts(self.max_n_dummy_nodes)
            if self.dummy_nodes_dist is not None: np.savetxt(os.path.join(info_dir, 'dummy_nodes_dist.txt'), self.dummy_nodes_dist.cpu().numpy())

            self.n_nodes = datamodule.node_counts()
            if self.n_nodes is not None and self.n_nodes.numel() > 0: self.max_n_nodes = len(self.n_nodes) - 1
            else: self.max_n_nodes = 0;
            if self.n_nodes is not None: np.savetxt(os.path.join(info_dir, 'n_counts.txt'), self.n_nodes.cpu().numpy())


            self.node_types = datamodule.node_types()
            if self.node_types is not None: np.savetxt(os.path.join(info_dir, 'atom_types.txt'), self.node_types.cpu().numpy())

            self.edge_types = datamodule.edge_counts()
            if self.edge_types is not None: np.savetxt(os.path.join(info_dir, 'edge_types.txt'), self.edge_types.cpu().numpy())
            
            if self.max_n_nodes is not None and self.max_n_nodes > 0 and "MIT" not in self.name: # Only for USPTO
                valencies = datamodule.valency_count(self.max_n_nodes)
                if valencies is not None:
                    np.savetxt(os.path.join(info_dir, 'valencies.txt'), valencies.cpu().numpy())
                    self.valency_distribution = valencies
            else: self.valency_distribution = None;

            if self.n_nodes is not None and self.n_nodes.numel() > 0: self.nodes_dist = utils.DistributionNodes(self.n_nodes)
            else: self.nodes_dist = None
        else:
            print(f"Warning (DatasetInfos for {self.name}): Dataloaders not available in datamodule when trying to compute distributions. Stats will be None.")
            self.dummy_nodes_dist = self.n_nodes = self.max_n_nodes = self.node_types = self.edge_types = self.valency_distribution = self.nodes_dist = None
    
    # ... (compute_input_output_dims method as it was in the last iteration)
    def compute_input_output_dims(self, datamodule, extra_features, domain_features, use_context):
        try:
            example_batch = next(iter(datamodule.train_dataloader()))
        except StopIteration:
            try:
                example_batch = next(iter(datamodule.val_dataloader()))
            except StopIteration:
                 num_atom_features = len(self.atom_encoder) if self.atom_encoder else 17 
                 num_bond_features = len(RetroBridgeDataset.bonds) + 1 if RetroBridgeDataset.bonds else 5 
                 self.input_dims = {'X': num_atom_features, 'E': num_bond_features, 'y': 1} 
                 self.output_dims = {'X': num_atom_features, 'E': num_bond_features, 'y': 0}
                 return 
        
        r_ex_dense, r_node_mask = utils.to_dense(
            example_batch.x, example_batch.edge_index, example_batch.edge_attr, example_batch.batch
        )
        p_ex_dense, p_node_mask = utils.to_dense(
            example_batch.p_x, example_batch.p_edge_index, example_batch.p_edge_attr, example_batch.batch
        )
        current_node_mask = p_node_mask 
        if not torch.all(r_node_mask == p_node_mask): pass

        p_example_data = {
            'X_t': p_ex_dense.X, 'E_t': p_ex_dense.E,
            'y_t': example_batch.y if hasattr(example_batch, 'y') else torch.empty(p_ex_dense.X.size(0), 0, device=p_ex_dense.X.device),
            'node_mask': current_node_mask
        }
        x_dim_base = example_batch.x.size(1) if hasattr(example_batch, 'x') and example_batch.x.numel() > 0 else len(self.atom_encoder)
        e_dim_base = example_batch.edge_attr.size(1) if hasattr(example_batch, 'edge_attr') and example_batch.edge_attr.numel() > 0 else (len(RetroBridgeDataset.bonds) + 1)
        y_dim_base = example_batch.y.size(1) if hasattr(example_batch, 'y') and example_batch.y.numel() > 0 else 0
        
        self.input_dims = {'X': x_dim_base, 'E': e_dim_base, 'y': y_dim_base + 1}
        ex_extra_feat = extra_features(p_example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)
        ex_extra_molecular_feat = domain_features(p_example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)
        if use_context:
            self.input_dims['X'] += x_dim_base 
            context_e_dim = example_batch.p_edge_attr.size(1) if hasattr(example_batch,'p_edge_attr') and example_batch.p_edge_attr.numel() > 0 else e_dim_base
            self.input_dims['E'] += context_e_dim
        self.output_dims = {'X': x_dim_base, 'E': e_dim_base, 'y': 0}


class RetroBridgeMITDatasetInfos(RetroBridgeDatasetInfos):
    atom_encoder = RetroBridgeMITDataset.types # Use MIT specific types
    atom_decoder = [k for k, v in sorted(RetroBridgeMITDataset.types.items(), key=lambda item: item[1])] # Derive
    max_n_dummy_nodes = RetroBridgeDatasetInfos.max_n_dummy_nodes # Inherit from parent static

    def __init__(self, datamodule: RetroBridgeMITDataModule): # Type hint
        # Call parent __init__ but it will run parent's init_attributes.
        # We need to specialize.
        # super().__init__(datamodule) # This calls RetroBridgeDatasetInfos.init_attributes
        
        # Instead, replicate parts of parent init and then call specialized init_attributes
        self.name = 'USPTO-MIT-RetroBridge' # Will be overridden by self.init_attributes
        self.input_dims = None; self.output_dims = None
        self.remove_h = True; self.max_weight = 1000
        self.possible_num_dummy_nodes = list(range(RetroBridgeMITDatasetInfos.max_n_dummy_nodes + 1))
        self.valencies = None; self.atom_weights = None # MIT does not use these from parent
        self.dummy_nodes_dist = None; self.n_nodes = None; self.max_n_nodes = None
        self.node_types = None; self.edge_types = None
        self.valency_distribution = None; self.nodes_dist = None
        self.num_reaction_classes = None; self.class_to_idx = None; self.idx_to_class = None

        self.init_attributes(datamodule) # Call MIT's version of init_attributes


    def init_attributes(self, datamodule: RetroBridgeMITDataModule): # This is specific to MIT
        self.name = 'USPTO-MIT-RetroBridge' 
        self.valencies = None # Not used for MIT in original
        self.atom_weights = None # Not used for MIT in original
        # Class info is not relevant for MIT data in this context
        self.num_reaction_classes = None 
        self.class_to_idx = None
        self.idx_to_class = None

        info_dir_base = getattr(datamodule, 'data_root', './data/uspto-mit') # Fallback for MIT data_root
        # MIT might have its own specific info_dir naming if needed
        if getattr(datamodule, 'extra_nodes', False):
            info_dir = os.path.join(info_dir_base, f'info_retrobridge_mit_extra_nodes') # MIT specific
        else:
            info_dir = os.path.join(info_dir_base, f'info_retrobridge_mit') # MIT specific
        os.makedirs(info_dir, exist_ok=True)
        
        # For MIT, stats are typically loaded if pre-computed, or computed.
        # The logic is similar to parent's _compute_and_save_distributions but might use different files or defaults.
        # Reusing parent's helper but it will save to MIT's info_dir.
        # The check `isinstance(datamodule, RetroBridgeDataModule) and not isinstance(datamodule, RetroBridgeMITDataModule)`
        # in the parent's init_attributes for class computation ensures it's skipped for MIT.
        
        # Copied loading/computing logic from parent's init_attributes, adapted for MIT context
        required_files_exist = all(
            os.path.exists(os.path.join(info_dir, fname))
            for fname in ['dummy_nodes_dist.txt', 'n_counts.txt', 'atom_types.txt', 'edge_types.txt']
        )
        # MIT does not typically compute/save valencies.txt in the original scripts.

        if getattr(datamodule, 'evaluation', False) and required_files_exist:
            # print(f"Info (DatasetInfos for {self.name}): Loading pre-computed statistics from {info_dir} (evaluation mode).")
            try:
                self.dummy_nodes_dist = torch.from_numpy(np.loadtxt(os.path.join(info_dir, 'dummy_nodes_dist.txt'))).float()
                self.n_nodes = torch.from_numpy(np.loadtxt(os.path.join(info_dir, 'n_counts.txt'))).float()
                if self.n_nodes.numel() > 0 : self.max_n_nodes = len(self.n_nodes) - 1
                else: self.max_n_nodes = 0; # print(f"Warning ({self.name}): Loaded n_nodes is empty.")
                self.node_types = torch.from_numpy(np.loadtxt(os.path.join(info_dir, 'atom_types.txt'))).float()
                self.edge_types = torch.from_numpy(np.loadtxt(os.path.join(info_dir, 'edge_types.txt'))).float()
                # No valency_distribution for MIT usually
                self.valency_distribution = None
                if self.n_nodes.numel() > 0 : self.nodes_dist = utils.DistributionNodes(self.n_nodes)
                else: self.nodes_dist = None
            except Exception as e:
                print(f"Error loading pre-computed statistics for {self.name} from {info_dir}: {e}. Attempting re-computation.")
                self._compute_and_save_distributions(datamodule, info_dir) # Fallback to compute
        else:
            # print(f"Info (DatasetInfos for {self.name}): Computing statistics for MIT (not evaluation mode or files missing).")
            self._compute_and_save_distributions(datamodule, info_dir) # Uses parent's helper