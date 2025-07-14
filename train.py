#python train.py --config configs/retrobridge.yaml --model RetroBridge --disable_wandb > training_log.txt 2>&1
import os
import argparse

from datetime import datetime

from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic
from src.data.retrobridge_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos
from src.data.retrobridge_dataset import RetroBridgeMITDataModule, RetroBridgeMITDatasetInfos # Keep for completeness
from src.features.extra_features import DummyExtraFeatures, ExtraFeatures
from src.features.extra_features_molecular import ExtraMolecularFeatures
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.metrics.molecular_metrics_discrete import DummyTrainMolecularMetricsDiscrete # Keep
from src.metrics.sampling_metrics import SamplingMolecularMetrics
from src.metrics.sampling_metrics import DummySamplingMolecularMetrics # Keep
from src.analysis.visualization import MolecularVisualization
from src.frameworks.markov_bridge import MarkovBridge
from src.frameworks.discrete_diffusion import DiscreteDiffusion
from src.frameworks.one_shot_model import OneShotModel

from pytorch_lightning import Trainer, callbacks, loggers
from pytorch_lightning.callbacks import ModelCheckpoint # Keep direct import if used

from pdb import set_trace

# ---- ADDITION START: Imports for plotting and optional logging ----
import matplotlib.pyplot as plt
import sys # For optional stdout/stderr redirection
# ---- ADDITION END ----


def find_last_checkpoint(checkpoints_dir):
    if not os.path.exists(checkpoints_dir) or not os.listdir(checkpoints_dir):
        return None
        
    if 'last.ckpt' in os.listdir(checkpoints_dir):
        last_ckpt_path = os.path.join(checkpoints_dir, 'last.ckpt')
        if os.path.exists(last_ckpt_path): 
            return last_ckpt_path

    top_5_checkpoints_dir = os.path.join(checkpoints_dir, 'top_5_accuracy') # Original name
    # ---- MODIFICATION START: Check if top_5_accuracy exists, otherwise check top_1_accuracy ----
    if not os.path.exists(top_5_checkpoints_dir) or not os.listdir(top_5_checkpoints_dir):
        top_5_checkpoints_dir = os.path.join(checkpoints_dir, 'top_1_accuracy') # Fallback
        if not os.path.exists(top_5_checkpoints_dir) or not os.listdir(top_5_checkpoints_dir):
            # If neither top_5 nor top_1 specific subdirs exist, check main dir for any ckpt
            any_ckpt_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt') and os.path.isfile(os.path.join(checkpoints_dir, f))]
            if any_ckpt_files:
                try:
                    latest_any_ckpt = max(any_ckpt_files, key=lambda f: os.path.getmtime(os.path.join(checkpoints_dir, f)))
                    return os.path.join(checkpoints_dir, latest_any_ckpt)
                except Exception: 
                    pass 
            return None
    # ---- MODIFICATION END ----


    epoch2fname = []
    for fname in os.listdir(top_5_checkpoints_dir): # Still uses top_5_checkpoints_dir which might now point to top_1
        if fname.endswith('.ckpt'):
            try:
                epoch_str = fname.split('_')[0] # Original logic might fail if filename format changes
                # ---- MODIFICATION START: More robust epoch parsing ----
                # Example filename: epoch=081-top_1_accuracy=0.345.ckpt or 081_top_1_accuracy=0.345.ckpt
                # We need to extract the number after 'epoch=' or the first number if no 'epoch='
                if '=' in epoch_str: 
                    epoch_val_str = epoch_str.split('=')[-1] # Get value after '='
                else: # Assume it's just the number
                    epoch_val_str = epoch_str
                
                # Further clean up if there are other non-numeric parts before the number
                # e.g. if epoch_str was "epoch081"
                epoch_val_str = ''.join(filter(str.isdigit, epoch_val_str))
                if epoch_val_str: # Ensure it's not empty after filtering
                    epoch_val = int(epoch_val_str)
                    epoch2fname.append((epoch_val, fname))
                # ---- MODIFICATION END ----
            except ValueError:
                # print(f"Warning: Could not parse epoch from checkpoint filename: {fname}")
                pass # Keep original behavior of just passing

    if not epoch2fname:
        return None
        
    latest_fname = max(epoch2fname, key=lambda t: t[0])[1]
    return os.path.join(top_5_checkpoints_dir, latest_fname)


# ---- ADDITION START: Plotting function ----
def plot_loss_curves(history, output_dir, experiment_name):
    epochs = range(1, len(history.get('retro_train_loss', [])) + 1)
    if not epochs:
        print("No loss history found to plot.")
        return

    plt.figure(figsize=(15, 10)) # Adjusted figure size for potentially more plots

    # Plot RetroBridge Losses
    plt.subplot(2, 1, 1) # Create a subplot for RetroBridge losses
    if history.get('retro_train_loss') and any(not isinstance(x, float) or not (x != x) for x in history['retro_train_loss']): # Check for non-NaN
        plt.plot(epochs, history['retro_train_loss'], 'b-', label='RetroBridge Train Loss')
    if history.get('retro_val_loss') and any(not isinstance(x, float) or not (x != x) for x in history['retro_val_loss']):
        plt.plot(epochs, history['retro_val_loss'], 'r-', label='RetroBridge Val Loss')
    plt.title(f'RetroBridge Task Losses for {experiment_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Classifier Losses and Accuracy (if available)
    # Check if there's any classifier data to plot
    has_classifier_data = (history.get('classifier_train_loss') and any(not isinstance(x, float) or not (x != x) for x in history['classifier_train_loss'])) or \
                          (history.get('classifier_val_loss') and any(not isinstance(x, float) or not (x != x) for x in history['classifier_val_loss'])) or \
                          (history.get('classifier_train_acc') and any(not isinstance(x, float) or not (x != x) for x in history['classifier_train_acc'])) or \
                          (history.get('classifier_val_acc') and any(not isinstance(x, float) or not (x != x) for x in history['classifier_val_acc']))

    if has_classifier_data:
        plt.subplot(2, 1, 2) # Create a second subplot for Classifier metrics
        ax_loss = plt.gca() # Get current axes for loss
        
        if history.get('classifier_train_loss') and any(not isinstance(x, float) or not (x != x) for x in history['classifier_train_loss']):
            ax_loss.plot(epochs, history['classifier_train_loss'], 'g-', label='Classifier Train Loss')
        if history.get('classifier_val_loss') and any(not isinstance(x, float) or not (x != x) for x in history['classifier_val_loss']):
            ax_loss.plot(epochs, history['classifier_val_loss'], 'm-', label='Classifier Val Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Classifier Loss')
        ax_loss.legend(loc='upper left')
        ax_loss.grid(True)

        # Create a twin axis for accuracy if data exists
        if (history.get('classifier_train_acc') and any(not isinstance(x, float) or not (x != x) for x in history['classifier_train_acc'])) or \
           (history.get('classifier_val_acc') and any(not isinstance(x, float) or not (x != x) for x in history['classifier_val_acc'])):
            ax_acc = ax_loss.twinx() # Create a twin y-axis sharing the same x-axis
            if history.get('classifier_train_acc') and any(not isinstance(x, float) or not (x != x) for x in history['classifier_train_acc']):
                ax_acc.plot(epochs, history['classifier_train_acc'], 'c--', label='Classifier Train Acc')
            if history.get('classifier_val_acc') and any(not isinstance(x, float) or not (x != x) for x in history['classifier_val_acc']):
                ax_acc.plot(epochs, history['classifier_val_acc'], 'y--', label='Classifier Val Acc')
            ax_acc.set_ylabel('Classifier Accuracy')
            ax_acc.legend(loc='upper right')
        
        plt.title(f'Auxiliary Classifier Metrics for {experiment_name}') # Title for the second subplot

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plot_path = os.path.join(output_dir, f'{experiment_name}_loss_curves.png')
    try:
        plt.savefig(plot_path)
        print(f"Loss curves saved to {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()
# ---- ADDITION END ----

# ---- ADDITION START: Optional Logger class for stdout/stderr redirection ----
class TeeLogger(object):
    def __init__(self, filename="Default.log", stream=sys.stdout, enabled=True):
        self.terminal = stream
        self.enabled = enabled
        if self.enabled:
            # Ensure the directory for the log file exists
            log_dir = os.path.dirname(filename)
            if log_dir and not os.path.exists(log_dir): # Check if log_dir is not empty
                os.makedirs(log_dir, exist_ok=True)
            self.log_file = open(filename, 'a') # Append mode
        else:
            self.log_file = None

    def write(self, message):
        self.terminal.write(message)
        if self.enabled and self.log_file:
            self.log_file.write(message)
            self.flush() # Ensure immediate write to file

    def flush(self):
        self.terminal.flush()
        if self.enabled and self.log_file:
            self.log_file.flush()

    def __getattr__(self, attr): # Handle other stdout/stderr attributes
        return getattr(self.terminal, attr)

    def close(self): # Method to close the log file
        if self.enabled and self.log_file:
            self.log_file.close()
# ---- ADDITION END ----


def main(args):
    # ---- ADDITION START: Setup TeeLogger for stdout/stderr ----
    # Determine if terminal logging to file is enabled (e.g., via a new config arg or default to True)
    enable_terminal_file_logging = getattr(args, 'log_terminal_output_to_file', False) # Default to False unless specified
    
    original_stdout = sys.stdout # Keep original stdout
    original_stderr = sys.stderr # Keep original stderr
    stdout_logger = None
    stderr_logger = None
    # ---- ADDITION END ----

    start_time = datetime.now().strftime('%d_%m_%H_%M_%S')
    run_name = f'{args.experiment_name}_{start_time}'
    experiment = run_name if args.resume is None else args.resume
    
    # ---- ADDITION START: Setup log file path for TeeLogger ----
    log_file_path = None
    if enable_terminal_file_logging:
        # Ensure args.logs directory exists before creating TeeLogger instance
        if not os.path.exists(args.logs):
            os.makedirs(args.logs, exist_ok=True)
        log_file_path = os.path.join(args.logs, f"{experiment}_terminal_output.txt")
        stdout_logger = TeeLogger(log_file_path, original_stdout, enabled=True)
        stderr_logger = TeeLogger(log_file_path, original_stderr, enabled=True) # Log stderr to the same file
        sys.stdout = stdout_logger
        sys.stderr = stderr_logger
    # ---- ADDITION END ----

    print(f'EXPERIMENT: {experiment}') # This will now also go to the log file if enabled

    data_root = os.path.join(args.data, args.dataset)
    checkpoints_dir = os.path.join(args.checkpoints, experiment)
    graphs_dir = os.path.join(args.logs, 'graphs', experiment)
    chains_dir = os.path.join(args.logs, 'chains', experiment)

    os.makedirs(args.logs, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(chains_dir, exist_ok=True)

    set_deterministic(args.seed)

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
        # ---- ADDITION START: Restore original stdout/stderr before raising error ----
        if enable_terminal_file_logging:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if stdout_logger: stdout_logger.close()
            # stderr_logger shares the same file, so closing stdout_logger is enough
        # ---- ADDITION END ----
        raise ValueError(f"Unsupported dataset: {args.dataset}. Expected 'uspto50k' or 'uspto-mit'.")

    dataset_infos = datamodule.dataset_infos
    if dataset_infos.max_n_nodes is None: 
        print("CRITICAL WARNING: dataset_infos.max_n_nodes is still None after datamodule initialization!")
        print("This indicates the initialization order fix in RetroBridgeDataModule might not have worked or there's another issue.")
    
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
    
    dataset_infos.compute_input_output_dims( 
        datamodule=datamodule,
        extra_features=extra_features_module, 
        domain_features=domain_features_module, 
        use_context=args.use_context, 
    )
    
    if args.dataset == "uspto-mit": 
        train_metrics_module = DummyTrainMolecularMetricsDiscrete() 
        sampling_metrics_module = DummySamplingMolecularMetrics()  
    else: 
        train_metrics_module = TrainMolecularMetricsDiscrete(dataset_infos)
        train_smiles_list = datamodule.train_smiles if hasattr(datamodule, 'train_smiles') else []
        if not train_smiles_list:
            print("Warning: train_smiles list is empty in datamodule. SamplingMolecularMetrics might not calculate novelty correctly.")
        sampling_metrics_module = SamplingMolecularMetrics(dataset_infos, train_smiles_list)
    
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
            intermediate_layer_idx_for_classification=getattr(args, 'intermediate_layer_idx_for_classification', 3),
            morgan_fp_radius=getattr(args, 'morgan_fp_radius', 2),
            morgan_fp_bits=getattr(args, 'morgan_fp_bits', 1024),
            classifier_hidden_dims=getattr(args, 'classifier_hidden_dims', [256, 128]),
            lambda_classification_loss=getattr(args, 'lambda_classification_loss', 0.1),
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
        # ---- ADDITION START: Restore original stdout/stderr before raising error ----
        if enable_terminal_file_logging:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if stdout_logger: stdout_logger.close()
        # ---- ADDITION END ----
        raise NotImplementedError(f"Model type '{args.model}' is not recognized in train.py.")


    top_1_checkpoints_dir = os.path.join(checkpoints_dir, 'top_1_accuracy')
    top_5_checkpoints_dir = os.path.join(checkpoints_dir, 'top_5_accuracy')
    os.makedirs(top_1_checkpoints_dir, exist_ok=True)
    os.makedirs(top_5_checkpoints_dir, exist_ok=True)

    checkpoint_callbacks_list = [ 
        callbacks.ModelCheckpoint(
            dirpath=top_1_checkpoints_dir,
            filename='{epoch:03d}_{top_1_accuracy:.3f}', 
            save_top_k=5,
            monitor=f'sampling_retro/top_1_accuracy', 
            mode='max',
            every_n_epochs=args.sample_every_val if hasattr(args, "sample_every_val") else 1, 
        ),
        callbacks.ModelCheckpoint(
            dirpath=top_5_checkpoints_dir,
            filename='{epoch:03d}_{top_5_accuracy:.3f}', 
            save_top_k=5,
            monitor=f'sampling_retro/top_5_accuracy', 
            mode='max',
            every_n_epochs=args.sample_every_val if hasattr(args, "sample_every_val") else 1, 
        )
    ]

    wandb_logger = None
    if not getattr(args, 'disable_wandb', False): 
        wandb_logger = loggers.WandbLogger(
            save_dir=args.logs,
            project='RetroBridge', 
            group=args.dataset,
            name=experiment,
            id=experiment, 
            resume='must' if args.resume is not None else 'allow',
            entity=getattr(args, 'wandb_entity', None), 
        )

    trainer = Trainer(
        max_epochs=args.n_epochs,
        logger=wandb_logger,
        callbacks=checkpoint_callbacks_list, 
        accelerator=args.device,
        devices=1, 
        num_sanity_val_steps=0,
        enable_progress_bar=args.enable_progress_bar, 
        log_every_n_steps=args.log_every_steps, 
    )
    
    last_checkpoint_path = None 
    if args.resume is None:
        print(f'No resume experiment name was passed – training from scratch or default latest.')
    else: 
        print(f"Attempting to resume experiment: {args.resume}")
        last_checkpoint_path = find_last_checkpoint(checkpoints_dir) 
        if last_checkpoint_path:
            print(f'Training will be resumed from the latest checkpoint: {last_checkpoint_path}')
        else:
            print(f'No checkpoint found for experiment {args.resume} in {checkpoints_dir}. Training from scratch.')


    print('Start training')
    
    # ---- ADDITION START: try-finally block to ensure loggers are closed ----
    try:
        trainer.fit(model=model_instance, datamodule=datamodule, ckpt_path=last_checkpoint_path)
        print('Training finished.')

        # Collect history data for plotting
        loss_history = {
            'retro_train_loss': model_instance.history_retro_train_loss if hasattr(model_instance, 'history_retro_train_loss') else [],
            'retro_val_loss': model_instance.history_retro_val_loss if hasattr(model_instance, 'history_retro_val_loss') else [],
            'classifier_train_loss': model_instance.history_classifier_train_loss if hasattr(model_instance, 'history_classifier_train_loss') else [],
            'classifier_val_loss': model_instance.history_classifier_val_loss if hasattr(model_instance, 'history_classifier_val_loss') else [],
            'classifier_train_acc': model_instance.history_classifier_train_acc if hasattr(model_instance, 'history_classifier_train_acc') else [],
            'classifier_val_acc': model_instance.history_classifier_val_acc if hasattr(model_instance, 'history_classifier_val_acc') else []
        }
        # Plotting function needs a directory to save images, using args.logs
        plot_loss_curves(loss_history, args.logs, experiment) 

    except Exception as e:
        print(f"An error occurred during training: {e}")
        # Optionally re-raise the exception if you want the script to exit with an error code
        # raise e 
    finally:
        # Restore original stdout and stderr and close log files
        if enable_terminal_file_logging:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if stdout_logger: stdout_logger.close()
            # stderr_logger shares the same file, so closing stdout_logger is enough
    # ---- ADDITION END ----


if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'), required=True)
    parser.add_argument('--model', type=str, required=True, help="Model type: RetroBridge, DiGress, OneShot")
    parser.add_argument('--disable_wandb', action='store_true', required=False, default=False, help="Disable W&B logging")
    # ---- ADDITION START: Argument for enabling terminal log to file ----
    parser.add_argument('--log_terminal_output_to_file', action='store_true', required=False, default=False, help="Enable logging terminal output to a file in the logs directory.")
    # ---- ADDITION END ----
    
    parsed_args = parser.parse_args()
    final_args = parse_yaml_config(parsed_args) 
    
    main(args=final_args)