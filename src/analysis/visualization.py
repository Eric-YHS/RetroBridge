import os
from pathlib import Path
import torch
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
import networkx as nx
import numpy as np
import rdkit.Chem
import wandb
import matplotlib.pyplot as plt

try:
    from .rdkit_functions import build_molecule 
except ImportError:
    try:
        from src.analysis.rdkit_functions import build_molecule
    except ImportError:
        print("CRITICAL ERROR in visualization.py: Could not import 'build_molecule'. Ensure src.analysis.rdkit_functions is accessible.")
        # Define a dummy build_molecule if import fails
        def build_molecule(*args, **kwargs): return Chem.RWMol(), 0 # Return empty Mol and 0 dummies

def create_dummy_conformer(molecule):
    conformer = AllChem.Conformer()
    for i in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(i, Point3D(0, 0, 0))

    molecule.AddConformer(conformer, assignId=True)
    return molecule


class MolecularVisualization:
    def __init__(self, dataset_infos):
        self.dataset_infos = dataset_infos

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol

    @staticmethod
    def _create_dummy_conformer(molecule):
        if molecule is None: return None
        try: # Add try-except for safety, especially around GetNumAtoms on potentially bad mol
            num_atoms = molecule.GetNumAtoms()
            if num_atoms == 0: return molecule # Nothing to do for empty mol
            conformer = AllChem.Conformer(num_atoms) 
        except Exception as e:
            print(f"Error creating conformer object for molecule: {e}")
            return molecule

        for i in range(num_atoms):
            conformer.SetAtomPosition(i, Point3D(0, 0, 0))
        
        existing_conf_ids = [conf.GetId() for conf in molecule.GetConformers()]
        for conf_id in existing_conf_ids:
            try:
                molecule.RemoveConformer(conf_id)
            except RuntimeError: # Catch error if conformer cannot be removed (e.g. invalid mol state)
                pass # Try to proceed
            
        try:
            molecule.AddConformer(conformer, assignId=True)
        except Exception as e:
            print(f"Error adding dummy conformer: {e}")
        return molecule

    def visualize_chain(self, path: str, nodes_list: np.ndarray, adjacency_matrix: np.ndarray, trainer=None):
        RDLogger.DisableLog('rdApp.*') 

        if not hasattr(self, 'dataset_infos') or not hasattr(self.dataset_infos, 'atom_decoder') or not self.dataset_infos.atom_decoder:
            print("Error (visualize_chain): self.dataset_infos or its atom_decoder is not properly set.")
            return None

        mols = []
        if not isinstance(nodes_list, np.ndarray) or nodes_list.ndim != 2 or nodes_list.shape[0] == 0:
            print(f"Warning (visualize_chain): Invalid nodes_list provided (shape: {nodes_list.shape if isinstance(nodes_list, np.ndarray) else type(nodes_list)}).")
            return mols 
        if not isinstance(adjacency_matrix, np.ndarray) or adjacency_matrix.ndim != 3 or adjacency_matrix.shape[0] != nodes_list.shape[0]:
            print(f"Warning (visualize_chain): Invalid adjacency_matrix or shape mismatch with nodes_list.")
            return mols


        for i in range(nodes_list.shape[0]): 
            current_nodes_indices = torch.from_numpy(nodes_list[i]).long() 
            current_adj_indices = torch.from_numpy(adjacency_matrix[i]).long() 
            
            try:
                mol, _ = build_molecule(
                    current_nodes_indices, 
                    current_adj_indices, 
                    self.dataset_infos.atom_decoder, 
                    return_n_dummy_atoms=True
                )
                mols.append(mol)
            except Exception as e:
                print(f"Error building molecule for frame {i}: {e}")
                mols.append(Chem.RWMol()) 

        if not any(mol is not None and mol.GetNumAtoms() > 0 for mol in mols):
            print("Warning (visualize_chain): No valid molecules were created from the chain data.")
            return mols

        final_molecule = None
        for mol_idx_rev in range(len(mols) - 1, -1, -1): 
            if mols[mol_idx_rev] is not None and mols[mol_idx_rev].GetNumAtoms() > 0:
                final_molecule = mols[mol_idx_rev]
                break
        
        if final_molecule is None:
            print("Warning (visualize_chain): No valid final molecule in chain for coord reference. Skipping GIF/Grid.")
            return mols

        try:
            AllChem.Compute2DCoords(final_molecule)
            if final_molecule.GetNumConformers() == 0: 
                final_molecule = self._create_dummy_conformer(final_molecule)
        except Exception as e:
            print(f'Error computing 2D coords for final_molecule: {e}. Using dummy conformer.')
            final_molecule = self._create_dummy_conformer(final_molecule)
        
        if final_molecule is None or final_molecule.GetNumConformers() == 0:
            print("Error (visualize_chain): Final molecule has no conformer. Cannot proceed with alignment.")
            return mols 

        coords = []
        try:
            final_conf = final_molecule.GetConformer() # Should exist now
            for i_atom_final in range(final_molecule.GetNumAtoms()):
                pos = final_conf.GetAtomPosition(i_atom_final)
                coords.append((pos.x, pos.y, pos.z))
        except Exception as e:
            print(f"Error getting atom positions from final_molecule's conformer: {e}")
            if final_molecule.GetNumAtoms() > 0:
                 coords = [(0,0,0)] * final_molecule.GetNumAtoms() # Fallback if atoms exist but positions fail
            else: # No atoms, no coords
                 return mols # Cannot proceed if final molecule has atoms but no coords extracted


        for i_mol, mol_intermediate in enumerate(mols): 
            if mol_intermediate is None or mol_intermediate.GetNumAtoms() == 0:
                mols[i_mol] = Chem.RWMol() 
                continue

            try:
                AllChem.Compute2DCoords(mol_intermediate)
                if mol_intermediate.GetNumConformers() == 0:
                    mol_intermediate = self._create_dummy_conformer(mol_intermediate)
            except Exception:
                mol_intermediate = self._create_dummy_conformer(mol_intermediate)

            if mol_intermediate is None or mol_intermediate.GetNumConformers() == 0:
                continue 

            num_atoms_intermediate = mol_intermediate.GetNumAtoms()
            intermediate_conf = mol_intermediate.GetConformer()

            for j_atom_intermediate in range(num_atoms_intermediate):
                if j_atom_intermediate < len(coords): 
                    x, y, z = coords[j_atom_intermediate]
                    try:
                        intermediate_conf.SetAtomPosition(j_atom_intermediate, Point3D(x, y, z))
                    except RuntimeError: 
                        break 
                else:
                    break 
            mols[i_mol] = mol_intermediate 

        os.makedirs(path, exist_ok=True) 
        save_paths = []
        num_valid_frames_for_gif = 0
        for frame_idx in range(len(mols)):
            if mols[frame_idx] is not None and mols[frame_idx].GetNumAtoms() > 0:
                file_name = os.path.join(path, f'frame_{frame_idx}.png')
                try:
                    Draw.MolToFile(mols[frame_idx], file_name, size=(300, 300), legend=f"Frame {frame_idx}")
                    save_paths.append(file_name)
                    num_valid_frames_for_gif += 1
                except Exception as e:
                    print(f"Error drawing molecule for frame {frame_idx} to file {file_name}: {e}")

        if not save_paths or num_valid_frames_for_gif == 0:
            print("Warning (visualize_chain): No valid frames saved for GIF.")
        else:
            try:
                imgs = [imageio.v2.imread(fn) for fn in save_paths] # Use imageio.v2.imread
                if imgs: 
                    gif_path = os.path.join(os.path.dirname(path), f'{Path(path).name}.gif')
                    imgs.extend([imgs[-1]] * 10) 
                    imageio.mimsave(gif_path, imgs, fps=5) 

                    if wandb.run:
                        try:
                            wandb.log({"chain": wandb.Video(gif_path, fps=5, format="gif")}, commit=True)
                        except Exception as e_wandb:
                            print(f"Error logging GIF to wandb: {e_wandb}")
                else:
                     print("Warning (visualize_chain): Failed to read saved frame images for GIF with imageio.v2.")
            except Exception as e_gif:
                print(f"Error creating or logging GIF: {e_gif}")

        try:
            valid_mols_for_grid = [m for m in mols if m and m.GetNumAtoms() > 0]
            if valid_mols_for_grid:
                mols_per_row_grid = min(10, len(valid_mols_for_grid))
                if mols_per_row_grid > 0: 
                    img = Draw.MolsToGridImage(valid_mols_for_grid, molsPerRow=mols_per_row_grid, subImgSize=(200, 200))
                    img_path = os.path.join(path, f'{Path(path).name}_grid_image.png') 
                    img.save(img_path)
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule for grid image.")
        except ValueError as ve: 
            print(f"ValueError creating grid image: {ve}")
        except Exception as e_grid:
            print(f"Error creating grid image: {e_grid}")
            
        return mols

    # ... (other methods of MolecularVisualization class, like 'visualize') ...
    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int, log='graph', prefix='', suffix=''):
        if not hasattr(self, 'dataset_infos') or not hasattr(self.dataset_infos, 'atom_decoder') or not self.dataset_infos.atom_decoder:
            print("Error (visualize): self.dataset_infos or its atom_decoder is not properly set.")
            return

        if not os.path.exists(path):
            os.makedirs(path)

        print(f"Visualizing {num_molecules_to_visualize} of {len(molecules)} to {path}")
        if num_molecules_to_visualize > len(molecules):
            # print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)
        
        actual_saved_count = 0
        for i in range(num_molecules_to_visualize):
            if i >= len(molecules): break # Should be caught by above, but defensive
            
            molecule_data = molecules[i] # This is [atom_types_tensor, edge_types_tensor]
            if not (isinstance(molecule_data, list) and len(molecule_data) == 2):
                print(f"Warning (visualize): Molecule data at index {i} is not in expected format [nodes, edges]. Skipping.")
                continue

            nodes_tensor, edges_tensor = molecule_data
            
            # Ensure they are tensors
            if not isinstance(nodes_tensor, torch.Tensor): nodes_tensor = torch.tensor(nodes_tensor, dtype=torch.long)
            if not isinstance(edges_tensor, torch.Tensor): edges_tensor = torch.tensor(edges_tensor, dtype=torch.long)


            file_path = os.path.join(path, f'{prefix}molecule_{i}{suffix}.png')
            try:
                # Use build_molecule as it directly takes discrete tensors
                mol, _ = build_molecule(
                    nodes_tensor.cpu(), # Ensure on CPU for rdkit
                    edges_tensor.cpu(), 
                    self.dataset_infos.atom_decoder,
                    return_n_dummy_atoms=True
                )
                if mol is not None and mol.GetNumAtoms() > 0:
                    Draw.MolToFile(mol, file_path)
                    actual_saved_count += 1
                    if wandb.run and log is not None:
                        # print(f"Saving {file_path} to wandb")
                        try:
                            wandb.log({log: wandb.Image(file_path)}, commit=False) # Usually commit=False inside a loop, commit once at end
                        except Exception as e_wandb:
                            print(f"Error logging image {file_path} to wandb: {e_wandb}")
                # else:
                    # print(f"Skipping saving molecule {i} to {file_path} (invalid or empty after build_molecule).")

            except rdkit.Chem.KekulizeException: # type: ignore
                print(f"Can't kekulize molecule {i} for saving to {file_path}.")
            except ValueError as ve: # E.g. from RDKit drawing
                print(f"ValueError drawing molecule {i} to {file_path}: {ve}")
            except Exception as e_draw:
                print(f"Generic error drawing molecule {i} to {file_path}: {e_draw}")
        
        if wandb.run and log is not None and actual_saved_count > 0: # Commit once after loop if any images logged
            try:
                wandb.log({}, commit=True) # Force commit
            except Exception as e_wandb_commit:
                 print(f"Error committing wandb log: {e_wandb_commit}")