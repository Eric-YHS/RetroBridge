# train.py

import os
import argparse

from datetime import datetime

from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic
from src.data.retrobridge_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos
from src.features.extra_features import DummyExtraFeatures, ExtraFeatures
from src.features.extra_features_molecular import ExtraMolecularFeatures
# <<< START MODIFICATION
# Import Dummy versions of metrics for faster training
from src.metrics.molecular_metrics_discrete import DummyTrainMolecularMetricsDiscrete
from src.metrics.sampling_metrics import DummySamplingMolecularMetrics
# We no longer need the full versions for this configuration
# from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
# from src.metrics.sampling_metrics import SamplingMolecularMetrics
# END MODIFICATION >>>
from src.analysis.visualization import MolecularVisualization
from src.frameworks.markov_bridge import MarkovBridge
from src.frameworks.discrete_diffusion import DiscreteDiffusion
from src.frameworks.one_shot_model import OneShotModel

from pytorch_lightning import Trainer, callbacks, loggers

from pdb import set_trace


def find_last_checkpoint(checkpoints_dir):
    if 'last.ckpt' in os.listdir(checkpoints_dir):
        return os.path.join(checkpoints_dir, 'last.ckpt')

    # Check for top_5_accuracy directory and checkpoints
    top_5_checkpoints_dir = os.path.join(checkpoints_dir, 'top_5_accuracy')
    if not os.path.exists(top_5_checkpoints_dir) or not os.listdir(top_5_checkpoints_dir):
        # Fallback to top_1_accuracy if top_5 is empty or does not exist
        top_1_checkpoints_dir = os.path.join(checkpoints_dir, 'top_1_accuracy')
        if not os.path.exists(top_1_checkpoints_dir) or not os.listdir(top_1_checkpoints_dir):
            print("No checkpoints found in top_1_accuracy or top_5_accuracy directories.")
            return None  # No checkpoint found

        # Use top_1 directory if top_5 is unavailable
        target_dir = top_1_checkpoints_dir
    else:
        target_dir = top_5_checkpoints_dir

    epoch2fname = [
        (int(fname.split('_')[0].split('=')[1]), fname)
        for fname in os.listdir(target_dir)
        if fname.endswith('.ckpt')
    ]
    if not epoch2fname:
        return None  # No valid checkpoint files found

    latest_fname = max(epoch2fname, key=lambda t: t[0])[1]
    return os.path.join(target_dir, latest_fname)


def main(args):
    start_time = datetime.now().strftime('%d_%m_%H_%M_%S')
    run_name = f'{args.experiment_name}_{start_time}'
    experiment = run_name if args.resume is None else args.resume
    print(f'EXPERIMENT: {experiment}')

    data_root = os.path.join(args.data, args.dataset)
    checkpoints_dir = os.path.join(args.checkpoints, experiment)
    graphs_dir = os.path.join(args.logs, 'graphs', experiment)
    chains_dir = os.path.join(args.logs, 'chains', experiment)

    os.makedirs(args.logs, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(chains_dir, exist_ok=True)

    set_deterministic(args.seed)

    datamodule = RetroBridgeDataModule(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        extra_nodes=args.extra_nodes,
        swap=args.swap,
        evaluation=False,
    )
    dataset_infos = RetroBridgeDatasetInfos(datamodule)

    extra_features = (
        ExtraFeatures(args.extra_features, dataset_info=dataset_infos)
        if args.extra_features is not None
        else DummyExtraFeatures()
    )
    domain_features = (
        ExtraMolecularFeatures(dataset_infos=dataset_infos)
        if args.extra_molecular_features
        else DummyExtraFeatures()
    )
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
        use_context=args.use_context,
    )

    # <<< START MODIFICATION
    # Use Dummy metrics to speed up training by skipping detailed metric calculations.
    # The main loss and top-k accuracy (for checkpointing) are unaffected.
    train_metrics = DummyTrainMolecularMetricsDiscrete()
    sampling_metrics = DummySamplingMolecularMetrics()
    # As requested, visualization is disabled.
    visualization_tools = None
    # END MODIFICATION >>>

    if args.model == 'RetroBridge':
        model = MarkovBridge(
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
            train_metrics=train_metrics,
            sampling_metrics=sampling_metrics,
            visualization_tools=visualization_tools,
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
        )
        # Manually set the feature extractor attributes on the model instance
        model.extra_features = extra_features
        model.domain_features = domain_features

    elif args.model == 'DiGress':
        model = DiscreteDiffusion(
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
            train_metrics=train_metrics,
            sampling_metrics=sampling_metrics,
            visualization_tools=visualization_tools,
            extra_features=extra_features,
            domain_features=domain_features,
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
        # Manually set the feature extractor attributes on the model instance
        model.extra_features = extra_features
        model.domain_features = domain_features

    elif args.model == 'OneShot':
        model = OneShotModel(
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
            train_metrics=train_metrics,
            sampling_metrics=sampling_metrics,
            visualization_tools=visualization_tools,
            extra_features=extra_features,
            domain_features=domain_features,
            log_every_steps=args.log_every_steps,
            sample_every_val=args.sample_every_val,
            samples_to_generate=args.samples_to_generate,
            samples_to_save=args.samples_to_save,
            samples_per_input=args.samples_per_input,
        )
        # Manually set the feature extractor attributes on the model instance
        model.extra_features = extra_features
        model.domain_features = domain_features

    top_1_checkpoints_dir = os.path.join(checkpoints_dir, 'top_1_accuracy')
    top_5_checkpoints_dir = os.path.join(checkpoints_dir, 'top_5_accuracy')
    os.makedirs(top_1_checkpoints_dir, exist_ok=True)
    os.makedirs(top_5_checkpoints_dir, exist_ok=True)

    checkpoint_callbacks = [
        callbacks.ModelCheckpoint(
            dirpath=top_1_checkpoints_dir,
            filename='{epoch:03d}_{top_1_accuracy:.3f}',
            save_top_k=5,
            monitor=f'top_1_accuracy',
            mode='max',
            every_n_epochs=args.sample_every_val,
        ),
        callbacks.ModelCheckpoint(
            dirpath=top_5_checkpoints_dir,
            filename='{epoch:03d}_{top_5_accuracy:.3f}',
            save_top_k=5,
            monitor=f'top_5_accuracy',
            mode='max',
            every_n_epochs=args.sample_every_val,
        )
    ]

    wandb_logger = None if args.disable_wandb else loggers.WandbLogger(
        save_dir=args.logs,
        project='RetroBridge',
        group=args.dataset,
        name=experiment,
        id=experiment,
        resume='must' if args.resume is not None else 'allow',
        entity=args.wandb_entity,
    )
    trainer = Trainer(
        max_epochs=args.n_epochs,
        logger=wandb_logger,
        callbacks=checkpoint_callbacks,
        accelerator=args.device,
        devices=1,
        num_sanity_val_steps=0,
        enable_progress_bar=args.enable_progress_bar,
        log_every_n_steps=args.log_every_steps,
    )

    if args.resume is None:
        last_checkpoint = None
        print(f'No checkpoint was passed – training from scratch')
    else:
        last_checkpoint = find_last_checkpoint(checkpoints_dir)
        print(
            f'Training will be resumed from the latest checkpoint {last_checkpoint}')

    print('Start training')
    trainer.fit(model=model,  datamodule=datamodule, ckpt_path=last_checkpoint)


if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=argparse.FileType(mode='r'), required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--disable_wandb', action='store_true',
                        required=False, default=False)
    main(args=parse_yaml_config(parser.parse_args()))
