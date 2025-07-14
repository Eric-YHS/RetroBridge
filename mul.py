import os
import argparse
import torch  # Added for GPU checks and matmul precision
from datetime import datetime

from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic
from src.data.retrobridge_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos
# Keep for completeness
from src.data.retrobridge_dataset import RetroBridgeMITDataModule, RetroBridgeMITDatasetInfos
from src.features.extra_features import DummyExtraFeatures, ExtraFeatures
from src.features.extra_features_molecular import ExtraMolecularFeatures
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.metrics.molecular_metrics_discrete import DummyTrainMolecularMetricsDiscrete  # Keep
from src.metrics.sampling_metrics import SamplingMolecularMetrics
from src.metrics.sampling_metrics import DummySamplingMolecularMetrics  # Keep
from src.analysis.visualization import MolecularVisualization
from src.frameworks.markov_bridge import MarkovBridge
from src.frameworks.discrete_diffusion import DiscreteDiffusion
from src.frameworks.one_shot_model import OneShotModel

from pytorch_lightning import Trainer, callbacks, loggers
from pytorch_lightning.callbacks import ModelCheckpoint  # Keep direct import if used

from pdb import set_trace

# ---- ADDITION START: Imports for plotting and optional logging ----
import matplotlib.pyplot as plt
import sys  # For optional stdout/stderr redirection
# ---- ADDITION END ----


def find_last_checkpoint(checkpoints_dir):
    if not os.path.exists(checkpoints_dir) or not os.listdir(checkpoints_dir):
        return None

    last_ckpt_path_main_dir = os.path.join(checkpoints_dir, 'last.ckpt')
    if os.path.exists(last_ckpt_path_main_dir) and os.path.isfile(last_ckpt_path_main_dir):
        return last_ckpt_path_main_dir

    potential_metric_dirs = ['top_1_accuracy', 'top_5_accuracy']
    best_metric_checkpoint_path = None
    highest_epoch_found = -1

    for metric_dir_name in potential_metric_dirs:
        metric_checkpoints_dir = os.path.join(checkpoints_dir, metric_dir_name)
        if os.path.exists(metric_checkpoints_dir) and os.path.isdir(metric_checkpoints_dir) and os.listdir(metric_checkpoints_dir):  # Added isdir check
            epoch2fname = []
            for fname in os.listdir(metric_checkpoints_dir):
                if fname.endswith('.ckpt') and os.path.isfile(os.path.join(metric_checkpoints_dir, fname)):
                    try:
                        epoch_val_str = ""
                        # Try to parse "epoch=XXX"
                        if "epoch=" in fname:
                            epoch_part = fname.split(
                                "epoch=")[1].split("-")[0].split("_")[0]
                            epoch_val_str = ''.join(
                                filter(str.isdigit, epoch_part))
                        else:  # Fallback: try to find the first sequence of digits
                            import re
                            match = re.search(
                                r'\d+', fname.split("-")[0].split("_")[0])
                            if match:
                                epoch_val_str = match.group(0)

                        if epoch_val_str:
                            epoch_val = int(epoch_val_str)
                            epoch2fname.append((epoch_val, fname))
                    except ValueError:
                        pass  # Ignore files where epoch cannot be parsed

            if epoch2fname:
                current_dir_latest_epoch, current_dir_latest_fname = max(
                    epoch2fname, key=lambda t: t[0])
                if current_dir_latest_epoch > highest_epoch_found:
                    highest_epoch_found = current_dir_latest_epoch
                    best_metric_checkpoint_path = os.path.join(
                        metric_checkpoints_dir, current_dir_latest_fname)
                elif current_dir_latest_epoch == highest_epoch_found and metric_dir_name == 'top_1_accuracy' and best_metric_checkpoint_path and 'top_5_accuracy' in best_metric_checkpoint_path:
                    best_metric_checkpoint_path = os.path.join(
                        metric_checkpoints_dir, current_dir_latest_fname)

    if best_metric_checkpoint_path:
        return best_metric_checkpoint_path

    any_ckpt_files = [
        f for f in os.listdir(checkpoints_dir)
        if f.endswith('.ckpt') and f != 'last.ckpt' and os.path.isfile(os.path.join(checkpoints_dir, f))
    ]
    if any_ckpt_files:
        try:
            latest_any_ckpt = max(any_ckpt_files, key=lambda f: os.path.getmtime(
                os.path.join(checkpoints_dir, f)))
            return os.path.join(checkpoints_dir, latest_any_ckpt)
        except Exception:
            pass

    return None


# ---- ADDITION START: Plotting function ----
def plot_loss_curves(history, output_dir, experiment_name):
    epochs = range(1, len(history.get('retro_train_loss', [])) + 1)
    if not epochs:
        pass

    has_any_data_to_plot = False
    for key in history:
        if history[key] and any(not (isinstance(x, float) and x != x) for x in history[key]):
            has_any_data_to_plot = True
            break

    if not has_any_data_to_plot:
        print("No valid data in history to plot loss curves.")
        return

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    has_retro_train_loss = history.get('retro_train_loss') and any(
        not (isinstance(x, float) and x != x) for x in history['retro_train_loss'])
    has_retro_val_loss = history.get('retro_val_loss') and any(
        not (isinstance(x, float) and x != x) for x in history['retro_val_loss'])

    if has_retro_train_loss:
        plt.plot(epochs, history['retro_train_loss'],
                 'b-', label='RetroBridge Train Loss')
    if has_retro_val_loss:
        plt.plot(epochs, history['retro_val_loss'],
                 'r-', label='RetroBridge Val Loss')

    if has_retro_train_loss or has_retro_val_loss:
        plt.title(f'RetroBridge Task Losses for {experiment_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    else:
        plt.title(f'No RetroBridge Task Losses Data for {experiment_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

    has_clf_train_loss = history.get('classifier_train_loss') and any(
        not (isinstance(x, float) and x != x) for x in history['classifier_train_loss'])
    has_clf_val_loss = history.get('classifier_val_loss') and any(
        not (isinstance(x, float) and x != x) for x in history['classifier_val_loss'])
    has_clf_train_acc = history.get('classifier_train_acc') and any(
        not (isinstance(x, float) and x != x) for x in history['classifier_train_acc'])
    has_clf_val_acc = history.get('classifier_val_acc') and any(
        not (isinstance(x, float) and x != x) for x in history['classifier_val_acc'])

    if has_clf_train_loss or has_clf_val_loss or has_clf_train_acc or has_clf_val_acc:
        plt.subplot(2, 1, 2)
        ax_loss = plt.gca()

        if has_clf_train_loss:
            ax_loss.plot(
                epochs, history['classifier_train_loss'], 'g-', label='Classifier Train Loss')
        if has_clf_val_loss:
            ax_loss.plot(
                epochs, history['classifier_val_loss'], 'm-', label='Classifier Val Loss')

        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Classifier Loss')
        if has_clf_train_loss or has_clf_val_loss:
            ax_loss.legend(loc='upper left')
        ax_loss.grid(True)

        if has_clf_train_acc or has_clf_val_acc:
            ax_acc = ax_loss.twinx()
            if has_clf_train_acc:
                ax_acc.plot(
                    epochs, history['classifier_train_acc'], 'c--', label='Classifier Train Acc')
            if has_clf_val_acc:
                ax_acc.plot(
                    epochs, history['classifier_val_acc'], 'y--', label='Classifier Val Acc')
            ax_acc.set_ylabel('Classifier Accuracy')
            if has_clf_train_acc or has_clf_val_acc:
                ax_acc.legend(loc='upper right')

        plt.title(f'Auxiliary Classifier Metrics for {experiment_name}')
    else:
        if len(plt.gcf().get_axes()) > 1:
            plt.subplot(2, 1, 2)
            plt.title(
                f'No Auxiliary Classifier Metrics Data for {experiment_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{experiment_name}_loss_curves.png')
    try:
        plt.savefig(plot_path)
        # Modified: Use global_rank for conditional print
        if int(os.environ.get("GLOBAL_RANK", 0)) == 0:
            print(f"Loss curves saved to {plot_path}")
    except Exception as e:
        if int(os.environ.get("GLOBAL_RANK", 0)) == 0:
            print(f"Error saving plot: {e}")
    plt.close()
# ---- ADDITION END ----

# ---- MODIFICATION START: TeeLogger made rank-aware ----


class TeeLogger(object):
    def __init__(self, filename="Default.log", stream=sys.stdout, enabled=True, rank=0):
        self.terminal = stream
        self.log_file = None
        self.rank = rank
        self.is_file_logging_enabled_for_this_rank = enabled and self.rank == 0

        if self.is_file_logging_enabled_for_this_rank:
            log_dir = os.path.dirname(filename)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            try:
                self.log_file = open(filename, 'a')  # Append mode
            except Exception as e:
                self.terminal.write(
                    f"Rank {self.rank} Error: Could not open log file {filename}: {e}\n")
                self.is_file_logging_enabled_for_this_rank = False

    def write(self, message):
        if self.rank == 0:
            self.terminal.write(message)
            if self.is_file_logging_enabled_for_this_rank and self.log_file:
                try:
                    self.log_file.write(message)
                    self.flush_file()
                except Exception as e:
                    self.terminal.write(
                        f"Rank {self.rank} Error: Could not write to log file: {e}\n")

    def flush(self):
        if self.rank == 0:
            self.terminal.flush()
            if self.is_file_logging_enabled_for_this_rank and self.log_file:
                self.flush_file()

    def flush_file(self):
        if self.is_file_logging_enabled_for_this_rank and self.log_file:
            try:
                self.log_file.flush()
            except Exception as e:
                self.terminal.write(
                    f"Rank {self.rank} Error: Could not flush log file: {e}\n")

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def close(self):
        if self.is_file_logging_enabled_for_this_rank and self.log_file:
            try:
                self.log_file.close()
            except Exception as e:
                self.terminal.write(
                    f"Rank {self.rank} Error: Could not close log file: {e}\n")
            self.log_file = None
# ---- MODIFICATION END ----


def main(args):
    if torch.cuda.is_available() and hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

    global_rank = int(os.environ.get("GLOBAL_RANK", 0))
    # Keep local_rank if needed elsewhere, though global_rank is primary for DDP logic
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    enable_terminal_file_logging = getattr(
        args, 'log_terminal_output_to_file', False)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_logger = None

    start_time = datetime.now().strftime('%d_%m_%H_%M_%S')
    run_name = f'{args.experiment_name}_{start_time}'
    experiment = run_name if args.resume is None else args.resume

    log_file_path = None
    if enable_terminal_file_logging:
        if global_rank == 0:  # Only rank 0 creates the log directory
            if not os.path.exists(args.logs):
                os.makedirs(args.logs, exist_ok=True)

        # Barrier to ensure directory is created before other ranks proceed if they also try to log (though they won't write with current TeeLogger)
        if world_size > 1:
            # Initialize a default process group if not already initialized, for the barrier
            if not torch.distributed.is_initialized():
                # Attempt to initialize a default process group. This might be tricky if
                # Lightning hasn't set up its DDP environment yet.
                # For simplicity, we assume Lightning handles DDP init before this part becomes critical for barrier.
                # If barrier is needed this early, manual DDP init might be required.
                # However, TeeLogger is rank-aware, so only rank 0 writes, reducing strict need for barrier here for dir.
                pass  # Let Lightning handle DDP init.

        shared_log_file_path = os.path.join(
            args.logs, f"{experiment}_terminal_output.txt")
        stdout_logger = TeeLogger(
            shared_log_file_path, original_stdout, enabled=True, rank=global_rank)
        sys.stdout = stdout_logger
        sys.stderr = stdout_logger

    if global_rank == 0:
        print(f'EXPERIMENT: {experiment}')

    data_root = os.path.join(args.data, args.dataset)
    checkpoints_dir = os.path.join(args.checkpoints, experiment)
    graphs_dir = os.path.join(args.logs, 'graphs', experiment)
    chains_dir = os.path.join(args.logs, 'chains', experiment)

    if global_rank == 0:
        os.makedirs(args.logs, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(graphs_dir, exist_ok=True)
        os.makedirs(chains_dir, exist_ok=True)

    set_deterministic(args.seed)

    trainer_accelerator = 'cpu'
    trainer_devices = 1
    trainer_strategy = None

    if hasattr(args, 'device'):
        if args.device == 'gpu':
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if global_rank == 0:
                    print(
                        f"CUDA is available. Number of GPUs detected: {num_gpus}")
                trainer_accelerator = 'gpu'
                trainer_devices = -1
                if world_size > 1:
                    trainer_strategy = 'ddp'
                    if global_rank == 0:
                        print(f"Using DDP strategy. World size: {world_size}.")
                elif num_gpus == 1:
                    trainer_devices = 1
                    if global_rank == 0:
                        print("Using 1 GPU.")
                elif num_gpus == 0:
                    if global_rank == 0:
                        print(
                            "Warning: CUDA reported available but 0 GPUs found. Falling back to CPU.")
                    trainer_accelerator = 'cpu'
                    trainer_devices = 1
            else:
                if global_rank == 0:
                    print(
                        "Warning: CUDA is not available, but 'gpu' was specified in config. Falling back to CPU.")
                trainer_accelerator = 'cpu'
                trainer_devices = 1
        elif args.device == 'cpu':
            if global_rank == 0:
                print("Using CPU as specified in config.")
            trainer_accelerator = 'cpu'
            trainer_devices = 1
        else:
            if global_rank == 0:
                print(
                    f"Warning: Unknown device '{args.device}' in config. Defaulting to CPU.")
            trainer_accelerator = 'cpu'
            trainer_devices = 1
    else:
        if global_rank == 0:
            print(
                "Warning: 'device' not specified in config. Attempting to use GPU if available, else CPU.")
        if torch.cuda.is_available():
            num_gpus_detected = torch.cuda.device_count()
            if global_rank == 0:
                print(
                    f"CUDA is available. Number of GPUs detected: {num_gpus_detected}")
            trainer_accelerator = 'gpu'
            trainer_devices = -1
            if world_size > 1:
                trainer_strategy = 'ddp'
                if global_rank == 0:
                    print(f"Using DDP strategy. World size: {world_size}.")
            elif num_gpus_detected == 1:
                trainer_devices = 1
                if global_rank == 0:
                    print("Using 1 GPU.")
            # Corrected indentation for the following elif and else
            elif num_gpus_detected == 0:
                if global_rank == 0:
                    print(
                        "Warning: CUDA reported available but 0 GPUs found. Falling back to CPU.")
                trainer_accelerator = 'cpu'
                trainer_devices = 1
        else:  # This else corresponds to `if torch.cuda.is_available():`
            if global_rank == 0:
                print("CUDA not available. Using CPU.")
            trainer_accelerator = 'cpu'
            trainer_devices = 1

    if trainer_strategy == 'ddp' and world_size > 0:
        if global_rank == 0 and args.num_workers > 0:
            print(
                f"DDP: Each of {world_size} processes will use {args.num_workers} dataloader workers.")

    if args.dataset == "uspto-mit":
        datamodule = RetroBridgeMITDataModule(
            data_root=data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            extra_nodes=args.extra_nodes,
            swap=args.swap,
            evaluation=False,
        )
    elif args.dataset == "uspto50k":
        datamodule = RetroBridgeDataModule(
            data_root=data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            extra_nodes=args.extra_nodes,
            swap=args.swap,
            evaluation=False,
        )
    else:
        if enable_terminal_file_logging and global_rank == 0:
            if stdout_logger:
                stdout_logger.close()
        if global_rank == 0 and enable_terminal_file_logging:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        raise ValueError(
            f"Unsupported dataset: {args.dataset}. Expected 'uspto50k' or 'uspto-mit'.")

    dataset_infos = datamodule.dataset_infos
    if dataset_infos.max_n_nodes is None and global_rank == 0:
        print("CRITICAL WARNING: dataset_infos.max_n_nodes is still None after datamodule initialization!")

    extra_features_module = (
        ExtraFeatures(args.extra_features, dataset_info=dataset_infos)
        if hasattr(args, 'extra_features') and args.extra_features is not None and args.extra_features != 'none'
        else DummyExtraFeatures()
    )
    domain_features_module = (
        ExtraMolecularFeatures(dataset_infos=dataset_infos)
        if hasattr(args, 'extra_molecular_features') and args.extra_molecular_features
        else DummyExtraFeatures()
    )

    if args.dataset == "uspto-mit":
        train_metrics_module = DummyTrainMolecularMetricsDiscrete()
        sampling_metrics_module = DummySamplingMolecularMetrics()
    else:
        train_metrics_module = TrainMolecularMetricsDiscrete(dataset_infos)
        train_smiles_list = datamodule.train_smiles if hasattr(
            datamodule, 'train_smiles') else []
        if not train_smiles_list and global_rank == 0:
            print("Warning: train_smiles list is empty in datamodule. SamplingMolecularMetrics might not calculate novelty correctly.")
        sampling_metrics_module = SamplingMolecularMetrics(
            dataset_infos, train_smiles_list)

    visualization_tools = MolecularVisualization(dataset_infos)

    model_instance = None
    if args.model == 'RetroBridge':
        model_instance = MarkovBridge(
            experiment_name=experiment,
            chains_dir=chains_dir,
            graphs_dir=graphs_dir,
            checkpoints_dir=checkpoints_dir,
            diffusion_steps=args.diffusion_steps,
            diffusion_noise_schedule=args.diffusion_noise_schedule,
            transition=args.transition,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_layers=args.n_layers,
            hidden_mlp_dims=args.hidden_mlp_dims,
            hidden_dims=args.hidden_dims,
            lambda_train=args.lambda_train,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics_module,
            sampling_metrics=sampling_metrics_module,
            visualization_tools=visualization_tools,
            extra_features=extra_features_module,
            domain_features=domain_features_module,
            use_context=args.use_context,
            log_every_steps=args.log_every_steps,
            sample_every_val=args.sample_every_val,
            samples_to_generate=args.samples_to_generate,
            samples_to_save=args.samples_to_save,
            samples_per_input=args.samples_per_input,
            chains_to_save=args.chains_to_save,
            number_chain_steps_to_save=args.number_chain_steps_to_save,
            fix_product_nodes=args.fix_product_nodes,
            loss_type=args.loss_type,
            intermediate_layer_idx_for_classification=getattr(
                args, 'intermediate_layer_idx_for_classification', 3),
            morgan_fp_radius=getattr(args, 'morgan_fp_radius', 2),
            morgan_fp_bits=getattr(args, 'morgan_fp_bits', 1024),
            classifier_hidden_dims=getattr(
                args, 'classifier_hidden_dims', [256, 128]),
            lambda_classification_loss=getattr(
                args, 'lambda_classification_loss', 0.1),
        )
    elif args.model == 'DiGress':
        model_instance = DiscreteDiffusion(
            experiment_name=experiment,
            chains_dir=chains_dir,
            graphs_dir=graphs_dir,
            checkpoints_dir=checkpoints_dir,
            diffusion_steps=args.diffusion_steps,
            diffusion_noise_schedule=args.diffusion_noise_schedule,
            transition=args.transition,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_layers=args.n_layers,
            hidden_mlp_dims=args.hidden_mlp_dims,
            hidden_dims=args.hidden_dims,
            lambda_train=args.lambda_train,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics_module,
            sampling_metrics=sampling_metrics_module,
            visualization_tools=visualization_tools,
            extra_features=extra_features_module,
            domain_features=domain_features_module,
            log_every_steps=args.log_every_steps,
            sample_every_val=args.sample_every_val,
            samples_to_generate=args.samples_to_generate,
            samples_to_save=args.samples_to_save,
            samples_per_input=args.samples_per_input,
            chains_to_save=args.chains_to_save,
            number_chain_steps_to_save=args.number_chain_steps_to_save,
            fix_product_nodes=args.fix_product_nodes,
            use_context=args.use_context,
        )
    elif args.model == 'OneShot':
        model_instance = OneShotModel(
            experiment_name=experiment,
            chains_dir=chains_dir,
            graphs_dir=graphs_dir,
            checkpoints_dir=checkpoints_dir,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_layers=args.n_layers,
            hidden_mlp_dims=args.hidden_mlp_dims,
            hidden_dims=args.hidden_dims,
            lambda_train=args.lambda_train,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics_module,
            sampling_metrics=sampling_metrics_module,
            visualization_tools=visualization_tools,
            extra_features=extra_features_module,
            domain_features=domain_features_module,
            log_every_steps=args.log_every_steps,
            sample_every_val=args.sample_every_val,
            samples_to_generate=args.samples_to_generate,
            samples_to_save=args.samples_to_save,
            samples_per_input=args.samples_per_input,
        )
    else:
        if enable_terminal_file_logging and global_rank == 0:
            if stdout_logger:
                stdout_logger.close()
        if global_rank == 0 and enable_terminal_file_logging:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        raise NotImplementedError(
            f"Model type '{args.model}' is not recognized in train.py.")

    top_1_checkpoints_dir = os.path.join(checkpoints_dir, 'top_1_accuracy')
    top_5_checkpoints_dir = os.path.join(checkpoints_dir, 'top_5_accuracy')
    if global_rank == 0:
        os.makedirs(top_1_checkpoints_dir, exist_ok=True)
        os.makedirs(top_5_checkpoints_dir, exist_ok=True)

    checkpoint_callbacks_list = [
        callbacks.ModelCheckpoint(
            dirpath=top_1_checkpoints_dir,
            filename='{epoch:03d}_{top_1_accuracy:.3f}',
            save_top_k=5,
            monitor=f'sampling_retro/top_1_accuracy',
            mode='max',
            every_n_epochs=args.sample_every_val if hasattr(
                args, "sample_every_val") else 1,
        ),
        callbacks.ModelCheckpoint(
            dirpath=top_5_checkpoints_dir,
            filename='{epoch:03d}_{top_5_accuracy:.3f}',
            save_top_k=5,
            monitor=f'sampling_retro/top_5_accuracy',
            mode='max',
            every_n_epochs=args.sample_every_val if hasattr(
                args, "sample_every_val") else 1,
        )
    ]

    wandb_logger = None
    if not getattr(args, 'disable_wandb', False) and global_rank == 0:
        wandb_logger = loggers.WandbLogger(
            save_dir=args.logs,
            project='RetroBridge',
            group=args.dataset,
            name=experiment,
            id=experiment,
            resume='must' if args.resume is not None else 'allow',
            entity=getattr(args, 'wandb_entity', None),
        )

    trainer_kwargs = {
        'max_epochs': args.n_epochs,
        'logger': wandb_logger,
        'callbacks': checkpoint_callbacks_list,
        'accelerator': trainer_accelerator,
        'devices': trainer_devices,
        'num_sanity_val_steps': 0,
        'enable_progress_bar': args.enable_progress_bar if global_rank == 0 else False,
        'log_every_n_steps': args.log_every_steps,
    }
    if trainer_strategy:
        trainer_kwargs['strategy'] = trainer_strategy

    uses_batch_norm = False
    if hasattr(model_instance, 'reaction_classifier') and model_instance.reaction_classifier is not None:
        for module in model_instance.reaction_classifier.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                uses_batch_norm = True
                break

    if trainer_strategy == 'ddp' and uses_batch_norm:  # Only add sync_batchnorm if DDP and BatchNorm is used
        trainer_kwargs['sync_batchnorm'] = True
        if global_rank == 0:
            print("sync_batchnorm=True enabled for DDP due to BatchNorm layers.")

    trainer = Trainer(**trainer_kwargs)

    last_checkpoint_path = None
    if args.resume is None:
        if global_rank == 0:
            print(
                f'No resume experiment name was passed – training from scratch or default latest.')
    else:
        if global_rank == 0:
            print(f"Attempting to resume experiment: {args.resume}")
        if global_rank == 0:
            last_checkpoint_path = find_last_checkpoint(checkpoints_dir)

        if world_size > 1:
            # Initialize DDP if not already done by Lightning for broadcast_object_list
            if not torch.distributed.is_initialized() and trainer_strategy == 'ddp':
                # This is a bit of a chicken-and-egg problem.
                # Lightning's Trainer usually handles DDP init.
                # If we need to broadcast before trainer.fit(), we might need to init manually.
                # However, Lightning's DDP strategy setup should handle checkpoint path synchronization.
                # Let's rely on Lightning to sync the ckpt_path if provided to trainer.fit().
                # So, only rank 0 needs to find it.
                pass

            # For DDP, Lightning handles checkpoint path internally if passed to fit.
            # No manual broadcast of path needed here if relying on Lightning's ckpt_path handling.
            # If manual broadcast is desired for some reason:
            # path_list = [last_checkpoint_path] if global_rank == 0 else [None]
            # torch.distributed.broadcast_object_list(path_list, src=0)
            # last_checkpoint_path = path_list[0]
            pass

        if last_checkpoint_path:
            if global_rank == 0:
                print(
                    f'Training will be resumed from the latest checkpoint: {last_checkpoint_path}')
        else:
            if global_rank == 0:
                print(
                    f'No checkpoint found for experiment {args.resume} in {checkpoints_dir}. Training from scratch.')

    if global_rank == 0:
        print('Start training')

    try:
        trainer.fit(model=model_instance, datamodule=datamodule,
                    ckpt_path=last_checkpoint_path)
        if global_rank == 0:
            print('Training finished.')
            loss_history = {
                'retro_train_loss': model_instance.history_retro_train_loss if hasattr(model_instance, 'history_retro_train_loss') else [],
                'retro_val_loss': model_instance.history_retro_val_loss if hasattr(model_instance, 'history_retro_val_loss') else [],
                'classifier_train_loss': model_instance.history_classifier_train_loss if hasattr(model_instance, 'history_classifier_train_loss') else [],
                'classifier_val_loss': model_instance.history_classifier_val_loss if hasattr(model_instance, 'history_classifier_val_loss') else [],
                'classifier_train_acc': model_instance.history_classifier_train_acc if hasattr(model_instance, 'history_classifier_train_acc') else [],
                'classifier_val_acc': model_instance.history_classifier_val_acc if hasattr(model_instance, 'history_classifier_val_acc') else []
            }
            plot_loss_curves(loss_history, args.logs, experiment)

    except Exception as e:
        if global_rank == 0:
            print(
                f"An error occurred during training on rank {global_rank}: {e}")
            import traceback
            traceback.print_exc()
    finally:
        if enable_terminal_file_logging:
            if stdout_logger:
                stdout_logger.close()

        if sys.stdout is stdout_logger:
            sys.stdout = original_stdout
        if sys.stderr is stdout_logger:
            sys.stderr = original_stderr


if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=argparse.FileType(mode='r'), required=True)
    parser.add_argument('--model', type=str, required=True,
                        help="Model type: RetroBridge, DiGress, OneShot")
    parser.add_argument('--disable_wandb', action='store_true',
                        required=False, default=False, help="Disable W&B logging")
    parser.add_argument('--log_terminal_output_to_file', action='store_true', required=False, default=False,
                        help="Enable logging terminal output (rank 0 only) to a file in the logs directory.")

    initial_parsed_args = parser.parse_args()
    final_args = parse_yaml_config(initial_parsed_args)

    main(args=final_args)
