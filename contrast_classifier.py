import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# =============================================================================
# 1. 定义模型架构 (直接从您的仓库文件中复制，确保脚本的独立性)
# =============================================================================


class ReactionClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim1: int = 256, hidden_dim2: int = 128, dropout_rate: float = 0.3):
        """
        A simple Multi-Layer Perceptron (MLP) classifier.
        (Copied from src.models.auxiliary_classifier.py for self-containment)

        Args:
            input_dim (int): Dimension of the input features.
            num_classes (int): Number of classes to classify (number of reaction types).
            hidden_dim1 (int): Dimension of the first hidden layer.
            hidden_dim2 (int): Dimension of the second hidden layer.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Note: BatchNorm is included but commented out to match the original implementation's
        # typical usage where it might be optional. You can enable it if needed.
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)  # Output logits
        return x

# =============================================================================
# 2. 数据处理 (Dataset 和 DataModule)
# =============================================================================


class FingerprintDataset(Dataset):
    """
    PyTorch Dataset to load reaction SMILES, convert them to Morgan fingerprints,
    and provide them for training the classifiers.
    """

    def __init__(self, csv_path: str, fp_radius: int, fp_bits: int):
        self.df = pd.read_csv(csv_path)
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits

        # Pre-calculate number of classes from the training data if it's a training set
        if 'train' in csv_path:
            # USPTO-50k classes are 1-10.
            self.num_classes = self.df['class'].max()
            print(
                f"Determined {self.num_classes} reaction classes from {csv_path}.")
        else:
            # For val/test, we assume 10 classes as per the dataset's standard.
            self.num_classes = 10

    def __len__(self):
        return len(self.df)

    def _smiles_to_fp(self, smiles_string: str) -> torch.Tensor:
        """Converts a SMILES string (potentially with multiple molecules) to a single fingerprint."""
        if not smiles_string or pd.isna(smiles_string):
            return torch.zeros(self.fp_bits, dtype=torch.float32)

        # Split reactants by '.' and generate a combined fingerprint
        mol_smiles_list = smiles_string.split('.')
        combined_fp = torch.zeros(self.fp_bits, dtype=torch.float32)

        for smi in mol_smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_bits)
                fp_tensor = torch.tensor(list(fp), dtype=torch.float32)
                # Combine fingerprints using bitwise OR logic (taking the max)
                combined_fp = torch.max(combined_fp, fp_tensor)
        return combined_fp

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        reaction_smiles = row['reactants>reagents>production']

        try:
            reactants_smi, _, product_smi = reaction_smiles.split('>')
        except ValueError:
            # Handle cases where the split doesn't work
            reactants_smi, product_smi = "", ""

        product_fp = self._smiles_to_fp(product_smi)
        reactants_fp = self._smiles_to_fp(reactants_smi)

        # Class labels are 1-10 in the CSV, convert to 0-9 for tensor indexing
        class_label = torch.tensor(row['class'] - 1, dtype=torch.long)

        return product_fp, reactants_fp, class_label


class ReactionDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule to handle the creation of data loaders.
    """

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, fp_radius: int, fp_bits: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.save_hyperparameters()

    def setup(self, stage=None):
        train_path = os.path.join(self.data_dir, 'uspto50k_train.csv')
        val_path = os.path.join(self.data_dir, 'uspto50k_val.csv')

        self.train_dataset = FingerprintDataset(
            train_path, self.fp_radius, self.fp_bits)
        self.val_dataset = FingerprintDataset(
            val_path, self.fp_radius, self.fp_bits)

        # Get num_classes from the training dataset instance
        self.num_classes = self.train_dataset.num_classes

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)


# =============================================================================
# 3. Pytorch Lightning Training System
# =============================================================================

class ComparativeClassifierSystem(pl.LightningModule):
    """
    A PyTorch Lightning module to train and evaluate the three comparative classifiers.
    """

    def __init__(self, input_dim_fp: int, num_classes: int, learning_rate: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters()

        # Experiment 1: Product fingerprint only
        self.classifier_prod = ReactionClassifier(
            input_dim=input_dim_fp, num_classes=num_classes)

        # Experiment 2: Reactant(s) fingerprint only
        self.classifier_react = ReactionClassifier(
            input_dim=input_dim_fp, num_classes=num_classes)

        # Experiment 3: Concatenated product and reactant(s) fingerprints
        self.classifier_concat = ReactionClassifier(
            input_dim=input_dim_fp * 2, num_classes=num_classes)

        # History tracking for final plot
        self.history = {
            'train_loss': {'prod': [], 'react': [], 'concat': []},
            'val_loss': {'prod': [], 'react': [], 'concat': []},
            'train_acc': {'prod': [], 'react': [], 'concat': []},
            'val_acc': {'prod': [], 'react': [], 'concat': []}
        }

    def _calculate_metrics(self, logits, labels):
        """Helper to compute loss and accuracy for a single classifier."""
        if len(logits) == 0:  # Handle empty batch in case of filtering
            return torch.tensor(0.0, device=self.device), 0.0
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        return loss, acc

    def _step(self, batch, batch_idx):
        """Common logic for training and validation steps."""
        prod_fp, react_fp, labels = batch

        # --- Classifier 1: Product FP ---
        logits_prod = self.classifier_prod(prod_fp)
        loss_prod, acc_prod = self._calculate_metrics(logits_prod, labels)

        # --- Classifier 2: Reactant FP ---
        logits_react = self.classifier_react(react_fp)
        loss_react, acc_react = self._calculate_metrics(logits_react, labels)

        # --- Classifier 3: Concatenated FP ---
        concat_fp = torch.cat([prod_fp, react_fp], dim=1)
        logits_concat = self.classifier_concat(concat_fp)
        loss_concat, acc_concat = self._calculate_metrics(
            logits_concat, labels)

        total_loss = loss_prod + loss_react + loss_concat

        results = {
            'total_loss': total_loss,
            'prod_loss': loss_prod.detach(),
            'react_loss': loss_react.detach(),
            'concat_loss': loss_concat.detach(),
            'prod_acc': acc_prod,
            'react_acc': acc_react,
            'concat_acc': acc_concat,
        }
        return results

    def training_step(self, batch, batch_idx):
        results = self._step(batch, batch_idx)

        # Log metrics for Pytorch Lightning's progress bar and logger
        self.log_dict({
            'train/loss_total': results['total_loss'],
            'train/loss_prod': results['prod_loss'],
            'train/loss_react': results['react_loss'],
            'train/loss_concat': results['concat_loss'],
        }, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return results['total_loss']

    def validation_step(self, batch, batch_idx):
        results = self._step(batch, batch_idx)

        self.log_dict({
            'val/loss_prod': results['prod_loss'],
            'val/loss_react': results['react_loss'],
            'val/loss_concat': results['concat_loss'],
            'val/acc_prod': results['prod_acc'],
            'val/acc_react': results['react_acc'],
            'val/acc_concat': results['concat_acc'],
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_train_epoch_end(self):
        # Retrieve epoch-level metrics logged by Lightning
        metrics = self.trainer.callback_metrics

        # It's better practice to aggregate metrics manually if you need them for plotting
        # For simplicity with Lightning, we'll just log and assume an aggregator is used.
        # But let's append to history here for the final plot.
        # We need to compute training accuracy here as it's not logged per step.
        # This requires iterating over the training dataloader again, which is slow.
        # A better way is to collect outputs from training_step.

        # For now, let's just print the validation results which are aggregated correctly.
        # Plotting will use the aggregated validation metrics.
        # We will manually calculate and store average training loss for plotting.
        pass

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        epoch = self.current_epoch + 1

        # Store metrics in history for plotting
        self.history['train_loss']['prod'].append(
            metrics.get('train/loss_prod_epoch', float('nan')))
        self.history['train_loss']['react'].append(
            metrics.get('train/loss_react_epoch', float('nan')))
        self.history['train_loss']['concat'].append(
            metrics.get('train/loss_concat_epoch', float('nan')))

        self.history['val_loss']['prod'].append(
            metrics['val/loss_prod'].item())
        self.history['val_loss']['react'].append(
            metrics['val/loss_react'].item())
        self.history['val_loss']['concat'].append(
            metrics['val/loss_concat'].item())

        self.history['val_acc']['prod'].append(metrics['val/acc_prod'].item())
        self.history['val_acc']['react'].append(
            metrics['val/acc_react'].item())
        self.history['val_acc']['concat'].append(
            metrics['val/acc_concat'].item())

        # Manually calculate and store training accuracy (by iterating over train loader)
        # This is done here to get a single, accurate value per epoch.
        train_accs = self._calculate_epoch_accuracy(
            self.trainer.datamodule.train_dataloader())
        self.history['train_acc']['prod'].append(train_accs['prod'])
        self.history['train_acc']['react'].append(train_accs['react'])
        self.history['train_acc']['concat'].append(train_accs['concat'])

        # Print summary
        print(f"\n--- Epoch {epoch} Summary ---")
        print(
            f"  Prod FP Classifier: Train Acc: {train_accs['prod']:.4f}, Val Acc: {metrics['val/acc_prod']:.4f}, Val Loss: {metrics['val/loss_prod']:.4f}")
        print(
            f"  React FP Classifier: Train Acc: {train_accs['react']:.4f}, Val Acc: {metrics['val/acc_react']:.4f}, Val Loss: {metrics['val/loss_react']:.4f}")
        print(
            f"  Concat FP Classifier: Train Acc: {train_accs['concat']:.4f}, Val Acc: {metrics['val/acc_concat']:.4f}, Val Loss: {metrics['val/loss_concat']:.4f}")
        print("------------------------\n")

    @torch.no_grad()
    def _calculate_epoch_accuracy(self, dataloader):
        """Calculates the accuracy over a full dataloader."""
        self.eval()  # Set model to evaluation mode

        all_labels = {'prod': [], 'react': [], 'concat': []}
        all_preds = {'prod': [], 'react': [], 'concat': []}

        for prod_fp, react_fp, labels in tqdm(dataloader, desc="Calculating train accuracy", leave=False):
            prod_fp = prod_fp.to(self.device)
            react_fp = react_fp.to(self.device)

            # Prod
            logits_prod = self.classifier_prod(prod_fp)
            all_labels['prod'].extend(labels.tolist())
            all_preds['prod'].extend(torch.argmax(
                logits_prod, dim=1).cpu().tolist())

            # React
            logits_react = self.classifier_react(react_fp)
            all_labels['react'].extend(labels.tolist())
            all_preds['react'].extend(torch.argmax(
                logits_react, dim=1).cpu().tolist())

            # Concat
            concat_fp = torch.cat([prod_fp, react_fp], dim=1)
            logits_concat = self.classifier_concat(concat_fp)
            all_labels['concat'].extend(labels.tolist())
            all_preds['concat'].extend(torch.argmax(
                logits_concat, dim=1).cpu().tolist())

        self.train()  # Set model back to training mode
        return {
            'prod': accuracy_score(all_labels['prod'], all_preds['prod']),
            'react': accuracy_score(all_labels['react'], all_preds['react']),
            'concat': accuracy_score(all_labels['concat'], all_preds['concat']),
        }

    def configure_optimizers(self):
        params = (
            list(self.classifier_prod.parameters()) +
            list(self.classifier_react.parameters()) +
            list(self.classifier_concat.parameters())
        )
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer


# =============================================================================
# 4. Plotting and Main Execution
# =============================================================================

def plot_results(history, output_dir):
    """Plots and saves the training history."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    fig.suptitle('Comparative Classifier Performance', fontsize=16)

    classifiers = ['prod', 'react', 'concat']
    titles = ['Product FP Only', 'Reactant(s) FP Only', 'Concatenated FPs']

    for i, (clf, title) in enumerate(zip(classifiers, titles)):
        epochs = range(1, len(history['train_loss'][clf]) + 1)

        # Plot Loss
        ax = axes[i, 0]
        ax.plot(epochs, history['train_loss'][clf], 'b-', label='Train Loss')
        ax.plot(epochs, history['val_loss'][clf],
                'r-', label='Validation Loss')
        ax.set_title(f'{title} - Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True)
        ax.legend()

        # Plot Accuracy
        ax = axes[i, 1]
        ax.plot(epochs, history['train_acc'][clf],
                'b--', label='Train Accuracy')
        ax.plot(epochs, history['val_acc'][clf],
                'r--', label='Validation Accuracy')
        ax.set_title(f'{title} - Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(output_dir, 'classifier_comparison_plots.png')
    plt.savefig(plot_path)
    print(f"\nTraining plots saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run comparative classifier experiments.")
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Directory containing the CSV files.')
    parser.add_argument('--output_dir', type=str, default='./classification_results',
                        help='Directory to save plots and logs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-5, help='Weight decay.')
    parser.add_argument('--fp_radius', type=int, default=2,
                        help='Morgan fingerprint radius.')
    parser.add_argument('--fp_bits', type=int, default=1024,
                        help='Morgan fingerprint bits.')
    parser.add_argument(
        '--gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use.')

    args = parser.parse_args()

    print("Starting comparative classifier training with the following configuration:")
    for k, v in vars(args).items():
        print(f"  - {k}: {v}")

    # --- Setup ---
    pl.seed_everything(42, workers=True)

    datamodule = ReactionDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fp_radius=args.fp_radius,
        fp_bits=args.fp_bits
    )
    # Important: call setup to determine num_classes
    datamodule.setup()

    model = ComparativeClassifierSystem(
        input_dim_fp=args.fp_bits,
        num_classes=datamodule.num_classes,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    # --- Trainer ---
    # Setup for multi-GPU training
    strategy = 'ddp_find_unused_parameters_true' if args.gpus > 1 else 'auto'

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=args.gpus,
        strategy=strategy,
        logger=pl.loggers.TensorBoardLogger(
            args.output_dir, name="comparison_logs"),
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=10)],
        deterministic=True,
    )

    # --- Run Training ---
    print(f"\nStarting training on {args.gpus} GPUs...")
    trainer.fit(model, datamodule)

    # --- Plot Results ---
    print("\nTraining finished. Generating plots...")
    plot_results(model.history, args.output_dir)


if __name__ == '__main__':
    main()
