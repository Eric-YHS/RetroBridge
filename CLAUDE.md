# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RetroBridge is a Markov Bridge Model implementation for retrosynthesis planning in chemistry. It models single-step retrosynthesis as a distribution learning problem in discrete state space, achieving state-of-the-art results on retrosynthesis benchmarks.

## Environment Setup

```bash
conda create --name retrobridge python=3.9 rdkit=2023.09.5 -c conda-forge -y
conda activate retrobridge
pip install -r requirements.txt
```

## Key Commands

### Training Models
- **RetroBridge**: `python train.py --config configs/retrobridge.yaml --model RetroBridge`
- **DiGress**: `python train.py --config configs/digress.yaml --model DiGress`  
- **ForwardBridge**: `python mit/train.py --config configs/forwardbridge.yaml`

### Inference/Sampling
- **Quick prediction**: `python predict.py --smiles "YOUR_MOLECULE_SMILES" --checkpoint models/retrobridge.ckpt`
- **Batch sampling**: `python sample.py --config configs/retrobridge.yaml --checkpoint models/retrobridge.ckpt --samples samples --model RetroBridge --mode test --n_samples 10 --n_steps 500`

### Model Downloads
```bash
mkdir -p models
wget https://zenodo.org/record/10688201/files/retrobridge.ckpt?download=1 -O models/retrobridge.ckpt
wget https://zenodo.org/record/10688201/files/digress.ckpt?download=1 -O models/digress.ckpt
wget https://zenodo.org/record/10688201/files/forwardbridge.ckpt?download=1 -O models/forwardbridge.ckpt
```

## Architecture

### Core Components

1. **Markov Bridge Framework** (`src/frameworks/markov_bridge.py`)
   - Main model class that implements the Markov bridge process
   - Handles forward/backward diffusion processes
   - Manages training loop and sampling procedures

2. **Graph Transformer** (`src/models/transformer_model.py`)
   - XEyTransformerLayer: Processes node features (X), edge features (E), and global features (y)
   - Multi-head attention mechanism for molecular graph processing
   - Separate feed-forward networks for each feature type

3. **Noise Scheduling** (`src/frameworks/noise_schedule.py`)
   - Implements cosine and other noise schedules for the diffusion process
   - InterpolationTransition for managing state transitions

4. **Dataset Management** (`src/data/retrobridge_dataset.py`)
   - Handles USPTO-50k dataset loading and preprocessing
   - Supports extra molecular features and graph augmentation

### Key Model Parameters (from configs)
- `diffusion_steps`: 500 (number of denoising steps)
- `n_layers`: 5 (transformer layers)
- `hidden_dims`: Node (256), Edge (64), Global (64)
- `batch_size`: 64
- `lr`: 0.0002

### Data Flow
1. Input product molecules are encoded as graphs with node/edge features
2. Forward diffusion adds noise progressively over T steps
3. Transformer model learns to reverse the diffusion process
4. Sampling generates reactant molecules by iterative denoising

## Important Considerations

- The model operates on molecular graphs, not SMILES strings directly
- `fix_product_nodes: True` ensures product atoms remain fixed during generation
- Round-trip evaluation requires Molecular Transformer (separate installation)
- Models checkpoint frequently - resume training with `--resume` flag