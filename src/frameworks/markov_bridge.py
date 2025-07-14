import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os

from src.data import utils
from src.frameworks.noise_schedule import InterpolationTransition, PredefinedNoiseScheduleDiscrete
from src.frameworks import diffusion_utils
from src.metrics.train_metrics import TrainLossDiscrete, TrainLossVLB
from src.metrics.sampling_metrics import compute_retrosynthesis_metrics # This was present
from src.models.transformer_model import GraphTransformer

# ---- MODIFICATION START ---- # (Original was here, kept for consistency)
from rdkit import Chem
from rdkit.Chem import AllChem
try:
    from src.models.auxiliary_classifier import ReactionClassifier
except ImportError:
    print("Warning: Could not import ReactionClassifier from src.models.auxiliary_classifier. Define it locally if needed.")
    ReactionClassifier = None 
# ---- MODIFICATION END ----


from sklearn.metrics import roc_auc_score # This was present
from tqdm import tqdm

from pdb import set_trace # This was present


class MarkovBridge(pl.LightningModule):
    def __init__(
            self,
            # Parameters that are typically simple types and will be saved as hparams
            experiment_name: str,
            diffusion_steps: int,
            diffusion_noise_schedule: str,
            lr: float,
            weight_decay: float,
            n_layers: int,
            hidden_mlp_dims: dict, 
            hidden_dims: dict,     
            lambda_train: list,    
            use_context: bool,
            log_every_steps: int,
            sample_every_val: int,
            samples_to_generate: int,
            samples_to_save: int,
            samples_per_input: int,
            chains_to_save: int,
            number_chain_steps_to_save: int,
            fix_product_nodes: bool,
            loss_type: str,
            intermediate_layer_idx_for_classification: int,
            morgan_fp_radius: int,
            morgan_fp_bits: int,
            classifier_hidden_dims: list, 
            lambda_classification_loss: float,
            
            # Parameters that are paths (strings, fine for hparams)
            chains_dir: str,
            graphs_dir: str,
            checkpoints_dir: str,
            
            # Parameters that might be simple (like a string for 'transition') or complex objects
            # If 'transition' is just a string config, it's fine. If it's an object, ignore it.
            # For now, assuming 'transition' from YAML is a string processed later.
            transition: str, # Name of the transition type from YAML

            # Complex object instances passed as arguments - these MUST be ignored by save_hyperparameters
            dataset_infos: object, # Actually an instance of RetroBridgeDatasetInfos
            train_metrics: nn.Module,
            sampling_metrics: nn.Module,
            visualization_tools: object, # Instance of MolecularVisualization
            extra_features: nn.Module,    # Instance of ExtraFeatures or DummyExtraFeatures
            domain_features: nn.Module,   # Instance of ExtraMolecularFeatures or DummyExtraFeatures
            # Add any other complex objects passed to __init__ here
    ):
        super().__init__()

        # Call save_hyperparameters() early.
        # It will automatically capture all arguments passed to __init__.
        # We then explicitly ignore the complex objects.
        self.save_hyperparameters(ignore=[
            'dataset_infos', 
            'train_metrics', 
            'sampling_metrics', 
            'visualization_tools',
            'extra_features',   
            'domain_features'
            # If 'transition' was an object instance, it should be ignored too.
            # For now, assuming 'transition' from YAML is a string like "interpolation"
            # and self.transition_model is constructed based on it later.
        ])
        # After this, self.hparams will store all other parameters.

        # --- Assign attributes to self, using self.hparams for saved hyperparameters ---
        
        # Validate loss_type early
        assert self.hparams.loss_type in ['cross_entropy', 'vlb'], \
            f"Invalid loss_type: {self.hparams.loss_type}. Must be 'cross_entropy' or 'vlb'."

        # These are complex objects, store them directly. They were ignored by save_hyperparameters.
        self.dataset_info = dataset_infos 
        self.train_metrics_module = train_metrics # Renamed to avoid conflict if self.train_metrics is a property
        self.sampling_metrics_module = sampling_metrics # Renamed
        self.visualization_tools = visualization_tools
        self.extra_features_module = extra_features 
        self.domain_features_module = domain_features

        # Access simple hparams via self.hparams
        self.name = self.hparams.experiment_name 
        self.chains_dir = self.hparams.chains_dir # chains_dir was a string, so it's in hparams
        self.graphs_dir = self.hparams.graphs_dir
        self.checkpoints_dir = self.hparams.checkpoints_dir

        self.model_dtype = torch.float32 # Static
        self.T = self.hparams.diffusion_steps
        
        # The 'transition' parameter from YAML (e.g., "interpolation") is stored in hparams.
        # The actual transition_model object is constructed below.
        # self.transition_param = self.hparams.transition # Storing the string name if needed

        self.lr = self.hparams.lr 
        self.weight_decay = self.hparams.weight_decay

        # Get dimensions from the dataset_infos object
        if not hasattr(self.dataset_info, 'input_dims') or self.dataset_info.input_dims is None or \
           not hasattr(self.dataset_info, 'output_dims') or self.dataset_info.output_dims is None:
            raise ValueError("dataset_infos is missing input_dims or output_dims. Ensure compute_input_output_dims was called.")

        input_dims = self.dataset_info.input_dims
        output_dims = self.dataset_info.output_dims
        nodes_dist = self.dataset_info.nodes_dist # May be None if not computed, handle downstream

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.train_loss = TrainLossDiscrete(self.hparams.lambda_train) if self.hparams.loss_type != 'vlb' else TrainLossVLB(self.hparams.lambda_train)
        self.val_loss = TrainLossDiscrete(self.hparams.lambda_train) if self.hparams.loss_type != 'vlb' else TrainLossVLB(self.hparams.lambda_train)
        
        self.use_context = self.hparams.use_context
        
        self.intermediate_layer_idx_for_classification = self.hparams.intermediate_layer_idx_for_classification
        self.morgan_fp_radius = self.hparams.morgan_fp_radius
        self.morgan_fp_bits = self.hparams.morgan_fp_bits
        self.lambda_classification_loss = self.hparams.lambda_classification_loss
        
        self.num_reaction_classes = getattr(self.dataset_info, 'num_reaction_classes', None)
        if self.num_reaction_classes is None or self.num_reaction_classes <= 0:
            print(f"Warning (MarkovBridge init): num_reaction_classes not properly set in dataset_info (value: {self.num_reaction_classes}).")
            self.num_reaction_classes = 10 # Fallback if not properly set
            print(f"Using fallback num_reaction_classes: {self.num_reaction_classes} for auxiliary classifier.")

        # Initialize the main model (GraphTransformer)
        self.model = GraphTransformer(
            n_layers=self.hparams.n_layers, 
            input_dims=input_dims, # Use locally derived input_dims
            hidden_mlp_dims=self.hparams.hidden_mlp_dims,
            hidden_dims=self.hparams.hidden_dims,
            output_dims=output_dims, # Use locally derived output_dims
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
            output_intermediate_y_at_layer=self.hparams.intermediate_layer_idx_for_classification # Pass this new param
        )
        
        # Initialize the auxiliary classifier
        self.reaction_classifier = None # Initialize as None
        if ReactionClassifier is not None and self.num_reaction_classes > 0:
            # Determine the input dimension for the classifier
            # It's the dimension of 'y' from the intermediate GraphTransformer layer + Morgan fingerprint bits
            transformer_y_dim = self.hparams.hidden_dims['dy'] # This should be the dimension of 'y' from GraphTransformer layers
            classifier_input_dim = transformer_y_dim + self.hparams.morgan_fp_bits
            h_dims_classifier = self.hparams.classifier_hidden_dims # This is a list from hparams
            
            # Basic validation for h_dims_classifier
            if not isinstance(h_dims_classifier, list) or not all(isinstance(d, int) for d in h_dims_classifier):
                print(f"Warning: classifier_hidden_dims is not a list of ints: {h_dims_classifier}. Using defaults [256, 128].")
                h_dims_classifier = [256, 128]

            self.reaction_classifier = ReactionClassifier(
                input_dim=classifier_input_dim,
                num_classes=self.num_reaction_classes,
                hidden_dim1=h_dims_classifier[0] if len(h_dims_classifier) > 0 else 256, # Default if list is too short
                hidden_dim2=h_dims_classifier[1] if len(h_dims_classifier) > 1 else 128  # Default if list is too short
            )
        else:
            if self.hparams.lambda_classification_loss > 0: # Only warn if loss weight is non-zero
                print("Warning: Auxiliary reaction classifier not initialized (ReactionClassifier class missing or num_reaction_classes invalid), but lambda_classification_loss > 0.")
        
        # Initialize noise schedule and transition model
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            noise_schedule=self.hparams.diffusion_noise_schedule,
            timesteps=self.hparams.diffusion_steps,
        )
        # MarkovBridge specifically uses InterpolationTransition. The 'transition' hparam might be redundant or for other models.
        self.transition_model = InterpolationTransition(
            x_classes=self.Xdim_output,
            e_classes=self.Edim_output,
            y_classes=self.ydim_output # y_classes is likely 0 for this model, but kept for consistency
        )

        # Other settings
        self.log_every_steps = self.hparams.log_every_steps
        self.sample_every_val = self.hparams.sample_every_val
        self.samples_to_generate = self.hparams.samples_to_generate
        self.samples_to_save = self.hparams.samples_to_save
        self.samples_per_input = self.hparams.samples_per_input
        self.chains_to_save = self.hparams.chains_to_save
        self.number_chain_steps_to_save = self.hparams.number_chain_steps_to_save
        self.val_counter = 0 # State variable, not hparam

        self.fix_product_nodes = self.hparams.fix_product_nodes
        self.loss_type = self.hparams.loss_type

        self.last_top_1_accuracy = 0.0
        self.last_top_5_accuracy = 0.0

        # ---- ADDITION START: History lists for plotting ----
        self.history_retro_train_loss = []
        self.history_retro_val_loss = []
        self.history_classifier_train_loss = []
        self.history_classifier_val_loss = []
        self.history_classifier_train_acc = []
        self.history_classifier_val_acc = []
        # ---- ADDITION END ----

        # ---- ADDITION START: Current epoch accumulators ----
        self.current_epoch_retro_train_loss_sum = 0.0
        self.current_epoch_classifier_train_loss_sum = 0.0
        self.current_epoch_classifier_train_acc_sum = 0.0
        self.num_train_batches_current_epoch = 0
        self.num_train_batches_with_valid_classification_targets = 0 # For accurate averaging of acc/loss

        self.current_epoch_retro_val_loss_sum = 0.0
        self.current_epoch_classifier_val_loss_sum = 0.0
        self.current_epoch_classifier_val_acc_sum = 0.0
        self.num_val_batches_current_epoch = 0
        self.num_val_batches_with_valid_classification_targets = 0 # For accurate averaging of acc/loss
        # ---- ADDITION END ----


    def configure_optimizers(self):
        # ---- MODIFICATION START ----
        params_to_optimize = list(self.model.parameters())
        if self.reaction_classifier is not None:
            params_to_optimize.extend(list(self.reaction_classifier.parameters()))
        
        return torch.optim.AdamW(
            params=params_to_optimize, # Optimize both models
            lr=self.lr, # Access via self
            weight_decay=self.weight_decay, # Access via self
            amsgrad=True,
        )
        # ---- MODIFICATION END ----

    def on_train_epoch_start(self):
        self.train_loss.reset()
        # ---- MODIFICATION START: Use renamed attribute ----
        if hasattr(self, 'train_metrics_module') and self.train_metrics_module is not None:
            self.train_metrics_module.reset()
        # ---- MODIFICATION END ----
        # ---- ADDITION START: Reset train epoch accumulators ----
        self.current_epoch_retro_train_loss_sum = 0.0
        self.current_epoch_classifier_train_loss_sum = 0.0
        self.current_epoch_classifier_train_acc_sum = 0.0
        self.num_train_batches_current_epoch = 0
        self.num_train_batches_with_valid_classification_targets = 0
        # ---- ADDITION END ----

    def process_and_forward(self, data, current_model_mode='retrobridge'): # Added mode for clarity
        # Getting graphs of reactants (target) and product (context)
        reactants, r_node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        reactants = reactants.mask(r_node_mask)

        product, p_node_mask = utils.to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product = product.mask(p_node_mask)

        if not torch.allclose(r_node_mask, p_node_mask): # Added check
             print("Warning in process_and_forward: reactant and product node masks differ.")
             # Using product node mask as the reference if they differ. This could be an issue.
             node_mask = p_node_mask
        else:
            node_mask = r_node_mask


        # Getting noisy data
        # Note that here products and reactants are swapped for MarkovBridge (X=product, X_T=reactants)
        noisy_data = self.apply_noise(
            X=product.X, E=product.E, y=product.y, # y is likely empty tensor from data
            X_T=reactants.X, E_T=reactants.E, y_T=reactants.y, # y_T also likely empty
            node_mask=node_mask,
        )

        # Computing extra features + context and making predictions
        context = product.clone() if self.use_context else None
        extra_data = self.compute_extra_data(noisy_data, context=context)
        
        # ---- MODIFICATION START ----
        # The self.model call will now return two values: (PlaceHolder_predictions, intermediate_y)
        # self.forward itself calls self.model
        pred_retro_placeholder, intermediate_y_for_classification = self.forward(noisy_data, extra_data, node_mask)
        # ---- MODIFICATION END ----

        # Masking unchanged part for the retro model's output
        if self.fix_product_nodes:
            fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
            modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
            # This assertion was inside training_step before, moved here as it's about product not pred.
            if not torch.all(fixed_nodes | modifiable_nodes): # Added check
                print("Warning: Not all nodes are either fixed or modifiable based on product's dummy atom indicator.")

            pred_retro_placeholder.X = pred_retro_placeholder.X * modifiable_nodes + product.X * fixed_nodes
            pred_retro_placeholder.X = pred_retro_placeholder.X * node_mask.unsqueeze(-1)
        
        # ---- MODIFICATION START ----
        return reactants, product, pred_retro_placeholder, node_mask, noisy_data, context, intermediate_y_for_classification
        # ---- MODIFICATION END ----

    def training_step(self, data, batch_idx): # Renamed 'i' to 'batch_idx'
        # 1. Process data and get model predictions
        reactants, product, pred_retro_placeholder, node_mask, noisy_data, context, intermediate_y = self.process_and_forward(data)
        # pred_retro_placeholder is the output for the main retrosynthesis task
        # intermediate_y is the output from the specified intermediate layer of GraphTransformer

        # 2. Calculate the original loss for the retrosynthesis task
        original_loss_val = torch.tensor(0.0, device=self.device) 
        
        if self.loss_type == 'vlb':
            # For VLB, train_loss.forward (called by compute_training_VLB) expects probability distributions
            original_loss_val = self.compute_training_VLB( # This method computes and returns the loss value
                reactants=reactants, 
                pred=pred_retro_placeholder, 
                node_mask=node_mask,
                noisy_data=noisy_data, # noisy_data contains t, z_t etc. needed for VLB loss terms
                batch_idx=batch_idx, # Pass batch_idx if needed for logging inside
            )
            # Ensure compute_training_VLB returns a tensor or a dict with 'loss'
            if isinstance(original_loss_val, dict) and 'loss' in original_loss_val:
                original_loss_val = original_loss_val['loss']
            elif not isinstance(original_loss_val, torch.Tensor):
                print(f"Warning: compute_training_VLB did not return a tensor or dict with 'loss'. Got: {type(original_loss_val)}")
                original_loss_val = torch.tensor(0.0, device=self.device)


        else: # loss_type == 'cross_entropy'
            # For CE, train_loss.forward expects logits and one-hot true labels
            original_loss_val = self.compute_training_CE_loss( # This method computes and returns the loss value
                reactants=reactants, 
                pred=pred_retro_placeholder, 
                batch_idx=batch_idx # Pass batch_idx if needed for logging inside
            )
            if not isinstance(original_loss_val, torch.Tensor):
                print(f"Warning: compute_training_CE_loss did not return a tensor. Got: {type(original_loss_val)}")
                original_loss_val = torch.tensor(0.0, device=self.device)


            # Update detailed training metrics (e.g., per-atom/bond CE)
            # This needs to be done *after* computing the main CE loss for the batch,
            # but *before* these metrics are computed for logging.
            # ---- MODIFICATION START: Use renamed attribute ----
            if hasattr(self, 'train_metrics_module') and self.train_metrics_module is not None:
            # ---- MODIFICATION END ----
                try:
                    self.train_metrics_module( # This is the forward() call, equivalent to self.train_metrics.update(...)
                        masked_pred_X=pred_retro_placeholder.X,
                        masked_pred_E=pred_retro_placeholder.E,
                        true_X=reactants.X,
                        true_E=reactants.E,
                    )
                except Exception as e:
                    print(f"Error during self.train_metrics_module update: {e}")
        
        # 3. Calculate auxiliary classification loss
        classification_loss = torch.tensor(0.0, device=self.device)
        classification_accuracy = 0.0 # For logging
        # ---- ADDITION START: For accurate averaging of acc/loss ----
        valid_indices_for_classification_loss = torch.empty(0, dtype=torch.bool, device=self.device) 
        # ---- ADDITION END ----


        if self.reaction_classifier is not None and intermediate_y is not None and self.lambda_classification_loss > 0:
            if not hasattr(data, 'p_smiles') or not hasattr(data, 'reaction_class'):
                # print("Warning (training_step): p_smiles or reaction_class missing in data batch. Skipping classification task.")
                pass # Reduce verbosity
            else:
                morgan_fingerprints_list = []
                for smi in data.p_smiles:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.morgan_fp_radius, nBits=self.morgan_fp_bits)
                        # Ensure fp is a list of floats/ints before converting to tensor
                        fp_list = list(map(int, fp.ToBitString())) # Example: convert to list of 0/1 ints
                        morgan_fingerprints_list.append(torch.tensor(fp_list, dtype=torch.float, device=self.device))
                    else:
                        morgan_fingerprints_list.append(torch.zeros(self.morgan_fp_bits, dtype=torch.float, device=self.device))
                
                if morgan_fingerprints_list: 
                    try:
                        morgan_fingerprints_tensor = torch.stack(morgan_fingerprints_list)
                        
                        if intermediate_y.size(0) == morgan_fingerprints_tensor.size(0):
                            classifier_input = torch.cat((intermediate_y, morgan_fingerprints_tensor), dim=1)
                            class_logits = self.reaction_classifier(classifier_input)
                            
                            target_classes = data.reaction_class.squeeze(-1) 
                            # ---- MODIFICATION START: Use renamed variable ----
                            valid_indices_for_classification_loss = (target_classes != -1) & (target_classes < self.num_reaction_classes) 
                            # ---- MODIFICATION END ----
                            
                            if valid_indices_for_classification_loss.any():
                                classification_loss = F.cross_entropy(class_logits[valid_indices_for_classification_loss], target_classes[valid_indices_for_classification_loss])
                                
                                with torch.no_grad(): # Calculate accuracy without affecting gradients
                                    preds_classes = torch.argmax(class_logits[valid_indices_for_classification_loss], dim=1)
                                    correct_classifications = (preds_classes == target_classes[valid_indices_for_classification_loss]).sum().item()
                                    # Avoid division by zero if valid_indices.sum() is 0, though valid_indices.any() checks this.
                                    num_valid_samples = valid_indices_for_classification_loss.sum().item()
                                    if num_valid_samples > 0:
                                        classification_accuracy = correct_classifications / num_valid_samples
                            # else:
                                # classification_loss remains 0.0 if no valid targets
                        # else:
                            # print(f"Warning (training_step): Batch size mismatch for classification. intermediate_y: {intermediate_y.shape}, morgan_fp: {morgan_fingerprints_tensor.shape}")
                    except RuntimeError as e: # Catch potential errors from torch.stack if list is malformed
                        print(f"Error stacking Morgan fingerprints or in classification: {e}")
                        classification_loss = torch.tensor(0.0, device=self.device) # Ensure it's a tensor
        
        # 4. Combine losses
        total_loss = original_loss_val + self.lambda_classification_loss * classification_loss

        # ---- ADDITION START: Accumulate training losses for epoch average ----
        self.current_epoch_retro_train_loss_sum += original_loss_val.item()
        self.num_train_batches_current_epoch += 1

        if self.reaction_classifier is not None and self.lambda_classification_loss > 0:
            # Only accumulate if there were valid targets for this batch's classification loss
            if valid_indices_for_classification_loss.any(): # Check the renamed variable
                 self.current_epoch_classifier_train_loss_sum += classification_loss.item()
                 self.current_epoch_classifier_train_acc_sum += classification_accuracy # Accuracy is already per-batch avg
                 self.num_train_batches_with_valid_classification_targets += 1
        # ---- ADDITION END ----
        
        # 5. Logging
        # Determine batch size for logging (can be tricky with PyG)
        current_batch_size = 1
        if hasattr(data, 'ptr') and data.ptr is not None: 
            current_batch_size = data.ptr.numel() - 1
        elif hasattr(data, 'batch') and data.batch is not None:
            current_batch_size = data.batch.max().item() + 1 # Number of unique graphs
        elif hasattr(data, 'x'):
            current_batch_size = data.x.size(0) # Fallback, might be num_nodes if not batched PyG
        if current_batch_size == 0 : current_batch_size = 1 # Avoid division by zero

        self.log('train_loss/original_loss', original_loss_val, on_step=True, on_epoch=True, batch_size=current_batch_size, sync_dist=True)
        if self.lambda_classification_loss > 0 and self.reaction_classifier is not None:
            self.log('train_loss/classification_loss', classification_loss, on_step=True, on_epoch=True, batch_size=current_batch_size, sync_dist=True)
            self.log('train_acc/classification_accuracy', classification_accuracy, on_step=False, on_epoch=True, batch_size=current_batch_size, sync_dist=True)
        self.log('train_loss/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=current_batch_size, sync_dist=True)

        # Log detailed metrics periodically
        if batch_idx > 0 and batch_idx % self.log_every_steps == 0 : # Avoid logging at step 0 before any updates for detailed metrics
            if self.loss_type == 'cross_entropy':
                 if hasattr(self, 'train_loss') and self.train_loss is not None and hasattr(self.train_loss, 'compute_metrics'):
                     computed_train_loss_metrics = self.train_loss.compute_metrics()
                     for metric_name, metric_val in computed_train_loss_metrics.items(): 
                         self.log(f'train_loss_CE/{metric_name}', metric_val, batch_size=current_batch_size, sync_dist=True)
                 
                 # ---- MODIFICATION START: Use renamed attribute ----
                 if hasattr(self, 'train_metrics_module') and self.train_metrics_module is not None and hasattr(self.train_metrics_module, 'compute_metrics'):
                 # ---- MODIFICATION END ----
                     computed_train_detailed_metrics = self.train_metrics_module.compute_metrics()
                     for metric_name, metric_val in computed_train_detailed_metrics.items():
                         self.log(f'train_detailed/{metric_name}', metric_val, batch_size=current_batch_size, sync_dist=True)
                 
                 # Reset after computing and logging
                 if hasattr(self, 'train_loss') and self.train_loss is not None and hasattr(self.train_loss, 'reset'): self.train_loss.reset() 
                 # ---- MODIFICATION START: Use renamed attribute ----
                 if hasattr(self, 'train_metrics_module') and self.train_metrics_module is not None and hasattr(self.train_metrics_module, 'reset'): self.train_metrics_module.reset() 
                 # ---- MODIFICATION END ----

            elif self.loss_type == 'vlb':
                 if hasattr(self, 'train_loss') and self.train_loss is not None and hasattr(self.train_loss, 'compute_metrics'):
                     computed_train_vlb_metrics = self.train_loss.compute_metrics()
                     for metric_name, metric_val in computed_train_vlb_metrics.items(): 
                         self.log(f'train_loss_VLB/{metric_name}', metric_val, batch_size=current_batch_size, sync_dist=True)
                     if hasattr(self.train_loss, 'reset'): self.train_loss.reset() 

        return total_loss

    # Renamed for clarity to distinguish from VLB version
    def compute_training_CE_loss(self, reactants, pred, batch_idx):
        loss = self.train_loss( # This is TrainLossDiscrete instance
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y, # pred.y is likely empty for retro model, loss_y will be 0
            true_X=reactants.X,
            true_E=reactants.E,
            true_y=reactants.y, # reactants.y also likely empty
        )
        # Original code logged detailed metrics here if batch_idx % self.log_every_steps == 0
        # Moved this detailed logging to the main training_step to avoid duplicate resets
        # and to log based on the combined loss.
        # Here we just return the loss value.
        return loss


    def compute_training_VLB(self, reactants, pred, node_mask, noisy_data, batch_idx): # Renamed i to batch_idx
        z_t = utils.PlaceHolder(X=noisy_data['X_t'], E=noisy_data['E_t'], y=noisy_data['y_t'])
        z_T_true = reactants # In MB, z_T is the reactant, z_0 is the product
        z_T_pred = pred      # Model predicts z_0 (product) based on z_t and z_T (reactant)
                             # Wait, for MB, model predicts z_0 (reactant) given z_t and z_T (product)
                             # So, z_T_true should be product, and z_0_pred is pred (reactant)
                             # Let's re-check the apply_noise and forward logic for MB
                             # apply_noise: X=product, X_T=reactant. Output noisy_data['X_t'] is noised product.
                             # forward: input noisy_data['X_t'] (noised product), context=product. Output pred (reactant).
                             # So, z_T_true here should be the target reactant. z_T_pred is the model's prediction of reactant.
        t = noisy_data['t'] # This 't' is model_perspective_t from apply_noise

        # q(z_s | z_t, z_0_true) where z_0_true is the true reactant
        # z_t is the noised product (from apply_noise)
        true_pX, true_pE = self.compute_q_zs_given_q_zt(z_t, z_T_true, node_mask, t=t)
        
        # p_theta(z_s | z_t) = sum_{z_0_pred} q(z_s | z_t, z_0_pred) p_theta(z_0_pred | z_t)
        # z_T_pred is the model's prediction of the reactant (z_0 in MB formulation)
        pred_pX, pred_pE = self.compute_p_zs_given_p_zt(z_t, z_T_pred, node_mask, t=t) # pred_pX/E are distributions

        loss = self.train_loss( # This is TrainLossVLB instance
            masked_pred_X=pred_pX, # These are now probability distributions for z_s
            masked_pred_E=pred_pE,
            true_X=true_pX,       # These are also probability distributions for z_s
            true_E=true_pE,
        )
        # Original code logged detailed metrics here. Moved to main training_step.
        return loss # Return just the loss value

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset() # For CE or VLB loss on validation data
        # ---- MODIFICATION START: Use renamed attribute ----
        if hasattr(self, 'sampling_metrics_module') and self.sampling_metrics_module is not None:
            self.sampling_metrics_module.reset()
        # ---- MODIFICATION END ----
        # ---- ADDITION START: Reset val epoch accumulators ----
        self.current_epoch_retro_val_loss_sum = 0.0
        self.current_epoch_classifier_val_loss_sum = 0.0
        self.current_epoch_classifier_val_acc_sum = 0.0
        self.num_val_batches_current_epoch = 0
        self.num_val_batches_with_valid_classification_targets = 0
        # ---- ADDITION END ----

    def validation_step(self, data, batch_idx): # Renamed i to batch_idx
        # ---- MODIFICATION START ----
        reactants, product, pred_retro_placeholder, node_mask, noisy_data, _, intermediate_y = self.process_and_forward(data)
        # ---- MODIFICATION END ----
        
        original_val_loss = torch.tensor(0.0, device=self.device)
        if self.loss_type == 'vlb':
            original_val_loss = self.compute_validation_VLB( # This needs to be implemented similarly to compute_training_VLB
                reactants=reactants,
                pred=pred_retro_placeholder,
                node_mask=node_mask,
                noisy_data=noisy_data,
                batch_idx=batch_idx,
            )
            if isinstance(original_val_loss, dict) and 'loss' in original_val_loss: # Similar handling as training
                original_val_loss = original_val_loss['loss']
        else:
            original_val_loss = self.compute_validation_CE_loss( # Needs to be implemented similarly
                reactants=reactants, 
                pred=pred_retro_placeholder, 
                batch_idx=batch_idx
            )
            if isinstance(original_val_loss, dict) and 'loss' in original_val_loss:
                original_val_loss = original_val_loss['loss']

        # ---- MODIFICATION START ----
        # Auxiliary Classification Loss for Validation
        classification_loss_val = torch.tensor(0.0, device=self.device)
        classification_accuracy_val = 0.0
        # ---- ADDITION START: For accurate averaging of acc/loss ----
        valid_indices_val_for_classification_loss = torch.empty(0, dtype=torch.bool, device=self.device) 
        # ---- ADDITION END ----


        if self.reaction_classifier is not None and intermediate_y is not None and self.lambda_classification_loss > 0:
            if not hasattr(data, 'p_smiles') or not hasattr(data, 'reaction_class'):
                 # print("Warning (val): p_smiles or reaction_class missing. Skipping classification.")
                 pass # Reduce verbosity for validation
            else:
                morgan_fingerprints_list_val = []
                for smi in data.p_smiles: # p_smiles from data batch
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.morgan_fp_radius, nBits=self.morgan_fp_bits)
                        # ---- MODIFICATION START: Corrected RDKit fingerprint to list conversion ----
                        morgan_fingerprints_list_val.append(torch.tensor(list(map(int, fp.ToBitString())), dtype=torch.float, device=self.device))
                        # ---- MODIFICATION END ----
                    else:
                        morgan_fingerprints_list_val.append(torch.zeros(self.morgan_fp_bits, dtype=torch.float, device=self.device))
                
                if morgan_fingerprints_list_val:
                    morgan_fingerprints_tensor_val = torch.stack(morgan_fingerprints_list_val)

                    if intermediate_y.size(0) != morgan_fingerprints_tensor_val.size(0):
                        # print(f"Warning (val): Batch size mismatch for classification. Skipping.")
                        pass
                    else:
                        classifier_input_val = torch.cat((intermediate_y, morgan_fingerprints_tensor_val), dim=1)
                        class_logits_val = self.reaction_classifier(classifier_input_val)
                        
                        target_classes_val = data.reaction_class.squeeze(-1)
                        # ---- MODIFICATION START: Use renamed variable ----
                        valid_indices_val_for_classification_loss = (target_classes_val != -1) & (target_classes_val < self.num_reaction_classes)
                        # ---- MODIFICATION END ----
                        
                        if valid_indices_val_for_classification_loss.any():
                            classification_loss_val = F.cross_entropy(class_logits_val[valid_indices_val_for_classification_loss], target_classes_val[valid_indices_val_for_classification_loss])
                            with torch.no_grad():
                                preds_classes_val = torch.argmax(class_logits_val[valid_indices_val_for_classification_loss], dim=1)
                                correct_classifications_val = (preds_classes_val == target_classes_val[valid_indices_val_for_classification_loss]).sum().item()
                                classification_accuracy_val = correct_classifications_val / valid_indices_val_for_classification_loss.sum().item()
                        else:
                            classification_loss_val = torch.tensor(0.0, device=self.device)
                else:
                    classification_loss_val = torch.tensor(0.0, device=self.device)


        total_val_loss = original_val_loss + self.lambda_classification_loss * classification_loss_val
        
        # ---- ADDITION START: Accumulate validation losses for epoch average ----
        self.current_epoch_retro_val_loss_sum += original_val_loss.item()
        self.num_val_batches_current_epoch += 1

        if self.reaction_classifier is not None and self.lambda_classification_loss > 0:
            # Only accumulate if there were valid targets for this batch's classification loss
            if valid_indices_val_for_classification_loss.any(): # Check the renamed variable
                self.current_epoch_classifier_val_loss_sum += classification_loss_val.item()
                self.current_epoch_classifier_val_acc_sum += classification_accuracy_val # Accuracy is already per-batch avg
                self.num_val_batches_with_valid_classification_targets += 1
        # ---- ADDITION END ----
        
        log_batch_size_val = data.ptr.numel() -1 if hasattr(data, 'ptr') and data.ptr is not None else (data.x.size(0) if hasattr(data, 'x') else 1) # Added check for data.ptr not None
        if log_batch_size_val == 0: log_batch_size_val = 1

        self.log('val_loss/original_loss', original_val_loss, on_step=False, on_epoch=True, batch_size=log_batch_size_val)
        if self.lambda_classification_loss > 0 and self.reaction_classifier is not None:
            self.log('val_loss/classification_loss', classification_loss_val, on_step=False, on_epoch=True, batch_size=log_batch_size_val)
            self.log('val_acc/classification_accuracy', classification_accuracy_val, on_step=False, on_epoch=True, batch_size=log_batch_size_val)
        self.log('val_loss/total_loss', total_val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=log_batch_size_val)
        
        # Log detailed CE metrics if using CE loss
        if self.loss_type == 'cross_entropy' and batch_idx % self.log_every_steps == 0: # From original train_step
            for metric_name, metric in self.val_loss.compute_metrics().items(): # val_loss should be CE here
                self.log(f'val_loss_CE/{metric_name}', metric, batch_size=log_batch_size_val)
            # self.val_loss.reset() # Reset after logging step-wise, or epoch-wise in on_validation_epoch_end
        elif self.loss_type == 'vlb' and batch_idx % self.log_every_steps == 0:
             for metric_name, metric in self.val_loss.compute_metrics().items(): # val_loss should be VLB here
                 self.log(f'val_loss_VLB/{metric_name}', metric, batch_size=log_batch_size_val)
            # self.val_loss.reset()


        return {'loss': total_val_loss} # Return combined loss for PyTorch Lightning
        # ---- MODIFICATION END ----

    # Renamed for clarity
    def compute_validation_CE_loss(self, reactants, pred, batch_idx):
        # self.val_loss is an instance of TrainLossDiscrete or TrainLossVLB
        # Call it directly
        loss = self.val_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=reactants.X,
            true_E=reactants.E,
            true_y=reactants.y,
        )
        # Original code had detailed logging here, which is now moved to validation_step
        return loss

    # Added for symmetry with training
    def compute_validation_VLB(self, reactants, pred, node_mask, noisy_data, batch_idx):
        z_t = utils.PlaceHolder(X=noisy_data['X_t'], E=noisy_data['E_t'], y=noisy_data['y_t'])
        z_T_true = reactants # Target for MB is reactant
        z_T_pred = pred      # Model predicts reactant
        t = noisy_data['t']  # model_perspective_t

        true_pX, true_pE = self.compute_q_zs_given_q_zt(z_t, z_T_true, node_mask, t=t)
        pred_pX, pred_pE = self.compute_p_zs_given_p_zt(z_t, z_T_pred, node_mask, t=t)

        loss = self.val_loss( # self.val_loss is instance of TrainLossVLB
            masked_pred_X=pred_pX,
            masked_pred_E=pred_pE,
            true_X=true_pX,
            true_E=true_pE,
        )
        return loss


    # ---- MODIFICATION START: Added on_train_epoch_end for printing training losses ----
    def on_train_epoch_end(self):
        # ---- ADDITION START: Calculate, print, and store average training losses ----
        epoch_num_display = self.current_epoch + 1 # current_epoch is 0-indexed
        max_epochs_display = self.trainer.max_epochs if self.trainer else 'N/A'
        
        if self.num_train_batches_current_epoch > 0:
            avg_retro_train_loss = self.current_epoch_retro_train_loss_sum / self.num_train_batches_current_epoch
            self.history_retro_train_loss.append(avg_retro_train_loss)
            print(f"Epoch {epoch_num_display}/{max_epochs_display} - RetroBridge Train Loss: {avg_retro_train_loss:.4f}", end="")

            if self.reaction_classifier is not None and self.lambda_classification_loss > 0:
                if self.num_train_batches_with_valid_classification_targets > 0:
                    avg_classifier_train_loss = self.current_epoch_classifier_train_loss_sum / self.num_train_batches_with_valid_classification_targets
                    avg_classifier_train_acc = self.current_epoch_classifier_train_acc_sum / self.num_train_batches_with_valid_classification_targets
                    self.history_classifier_train_loss.append(avg_classifier_train_loss)
                    self.history_classifier_train_acc.append(avg_classifier_train_acc)
                    print(f", Classifier Train Loss: {avg_classifier_train_loss:.4f}, Classifier Train Acc: {avg_classifier_train_acc:.4f}", end="")
                else: # If no valid classification targets were encountered in the epoch
                    self.history_classifier_train_loss.append(float('nan')) # Or 0.0, or skip
                    self.history_classifier_train_acc.append(float('nan'))  # Or 0.0, or skip
                    print(f", Classifier Train Loss: N/A, Classifier Train Acc: N/A", end="")
            print() # Newline after all train losses for the epoch
        # ---- ADDITION END ----
    # ---- MODIFICATION END ----

# 文件路径: src/frameworks/markov_bridge.py
# 在 MarkovBridge 类中

# vvvvvvvvvvvv 从这里开始替换 vvvvvvvvvvvv
    def on_validation_epoch_end(self):
        # 1. 聚合和打印当前验证周期的平均损失 (这部分逻辑来自你之前的代码)
        epoch_num_display = self.current_epoch + 1
        max_epochs_display = self.trainer.max_epochs if self.trainer else 'N/A'

        if self.num_val_batches_current_epoch > 0:
            avg_retro_val_loss = self.current_epoch_retro_val_loss_sum / self.num_val_batches_current_epoch
            self.history_retro_val_loss.append(avg_retro_val_loss)
            print(f"Epoch {epoch_num_display}/{max_epochs_display} - RetroBridge Val Loss: {avg_retro_val_loss:.4f}", end="")

            if self.reaction_classifier is not None and self.lambda_classification_loss > 0:
                if self.num_val_batches_with_valid_classification_targets > 0:
                    avg_classifier_val_loss = self.current_epoch_classifier_val_loss_sum / self.num_val_batches_with_valid_classification_targets
                    avg_classifier_val_acc = self.current_epoch_classifier_val_acc_sum / self.num_val_batches_with_valid_classification_targets
                    self.history_classifier_val_loss.append(avg_classifier_val_loss)
                    self.history_classifier_val_acc.append(avg_classifier_val_acc)
                    print(f", Classifier Val Loss: {avg_classifier_val_loss:.4f}, Classifier Val Acc: {avg_classifier_val_acc:.4f}", end="")
                else:
                    self.history_classifier_val_loss.append(float('nan'))
                    self.history_classifier_val_acc.append(float('nan'))
                    print(f", Classifier Val Loss: N/A, Classifier Val Acc: N/A", end="")
            print() # 换行

        # 2. 重置验证损失计算器
        self.val_loss.reset()

        # 3. 核心逻辑：处理采样和指标记录
        self.val_counter += 1
        run_sampling_this_epoch = (self.val_counter % self.sample_every_val == 0)

        if run_sampling_this_epoch:
            # 如果是采样周期，正常运行 sample()
            print(f"--- Running sampling for validation epoch {self.current_epoch + 1} ---")
            if hasattr(self, 'sample') and callable(self.sample):
                self.sample()
                # sample() 方法内部会通过 self.log() 记录新的准确率。
                # 我们在这里从 trainer 的指标字典中获取这些新值，并保存起来供后续周期使用。
                # 使用 .get(key, default_value) 来安全地获取指标
                self.last_top_1_accuracy = self.trainer.callback_metrics.get('sampling_retro/top_1_accuracy', self.last_top_1_accuracy).item()
                self.last_top_5_accuracy = self.trainer.callback_metrics.get('sampling_retro/top_5_accuracy', self.last_top_5_accuracy).item()
                print(f"--- Sampling finished. New accuracies logged: Top-1={self.last_top_1_accuracy:.4f}, Top-5={self.last_top_5_accuracy:.4f} ---")
            else:
                print("Warning: self.sample method not found or not callable.")
        else:
            # 如果不是采样周期，我们手动记录上一次的准确率值。
            # 这确保了 ModelCheckpoint 回调函数总能找到它要监控的 key。
            # 使用 on_step=False, on_epoch=True, sync_dist=True 与 self.log 在采样中的默认行为保持一致。
            self.log('sampling_retro/top_1_accuracy', self.last_top_1_accuracy, on_step=False, on_epoch=True, sync_dist=True)
            self.log('sampling_retro/top_5_accuracy', self.last_top_5_accuracy, on_step=False, on_epoch=True, sync_dist=True)

        # 4. ModelCheckpoint 回调会自动运行，我们不需要在这里手动保存检查点。
        #    它会根据我们刚刚记录的（新的或旧的）'sampling_retro/top_1_accuracy' 等指标来决定是否保存。
    # ^^^^^^^^^^^^^^ 到这里结束替换 ^^^^^^^^^^^^^^


    def sample(self): # This method remains largely the same for generating main task samples
        samples_left_to_generate = self.samples_to_generate
        samples_left_to_save = self.samples_to_save
        chains_left_to_save = self.chains_to_save

        samples = []
        grouped_samples = []
        grouped_scores = [] # Scores are currently [0] * len(molecule_list) from sample_batch
        ground_truth = []

        ident = 0
        # print(f'Sampling epoch={self.current_epoch}') # self.current_epoch might not be available if not in PL trainer context.

        # ---- MODIFICATION START ----
        # Ensure dataloader is available
        if not hasattr(self.trainer, 'datamodule') or self.trainer.datamodule is None:
            print("Error: Trainer datamodule not available for sampling.")
            return
        val_dataloader = self.trainer.datamodule.val_dataloader()
        if val_dataloader is None:
            print("Error: Validation dataloader not available for sampling.")
            return
        
        # Progress bar total calculation
        total_batches = float('inf') # Default if samples_left_to_generate is not positive
        if hasattr(val_dataloader, 'batch_size') and val_dataloader.batch_size is not None and val_dataloader.batch_size > 0 and samples_left_to_generate > 0:
            total_batches = samples_left_to_generate // val_dataloader.batch_size
            if samples_left_to_generate % val_dataloader.batch_size != 0:
                 total_batches +=1
        elif samples_left_to_generate > 0 : # If batch_size is unknown, iterate through dataloader once
            total_batches = len(val_dataloader) if hasattr(val_dataloader, '__len__') else float('inf')


        pbar_desc = f'Sampling epoch'
        if hasattr(self, 'current_epoch'):
            pbar_desc = f'Sampling epoch={self.current_epoch + 1}' # Use 1-indexed for display
        
        # Create tqdm iterable only if total_batches is finite and positive
        if total_batches != float('inf') and total_batches > 0 : # Add total_batches > 0
            dataloader_iterable = tqdm(val_dataloader, total=int(total_batches), desc=pbar_desc)
        else:
            dataloader_iterable = val_dataloader # Iterate without progress bar if total is unknown or zero
        # ---- MODIFICATION END ----

        for data in dataloader_iterable: # Use the iterable
            if samples_left_to_generate <= 0:
                break

            data = data.to(self.device)
            # ---- MODIFICATION START ----
            # bs = len(data.batch.unique()) if hasattr(data,'batch') and data.batch is not None else data.x.size(0) # Fallback if no batch attr
            if hasattr(data, 'ptr') and data.ptr is not None: # ptr indicates batch size for PyG
                bs = data.ptr.numel() - 1
            elif hasattr(data, 'batch') and data.batch is not None:
                bs = len(data.batch.unique())
            else: # Fallback for non-PyG data or single graph
                bs = data.x.size(0) if hasattr(data, 'x') else 1
            if bs == 0: continue # Skip empty batches
            # ---- MODIFICATION END ----

            to_generate = bs
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            batch_groups_reactants = [] # Renamed for clarity
            batch_scores_reactants = [] # Renamed

            for s_idx in range(self.samples_per_input): # Renamed sample_idx to s_idx
                # ---- MODIFICATION START ----
                # sample_batch now returns (molecule_list, true_molecule_list, products_list, scores, nll, ell)
                # We need to ensure all these are handled or ignored if not needed for this specific task.
                # The original `sample_batch` in `MarkovBridge` already has this signature.
                mol_list, true_mol_list, prod_list, current_scores, _, _ = self.sample_batch( # Keep _ for nll, ell if not used here
                    data=data,
                    batch_id=ident,
                    batch_size=to_generate,
                    save_final=to_save, # This controls visualization inside sample_batch
                    keep_chain=chains_save, # This also for visualization
                    number_chain_steps_to_save=self.number_chain_steps_to_save,
                    sample_idx=s_idx, # Original parameter name
                    # save_true_reactants=True, # Default in original was True
                    # use_one_hot=False, # Default in original sample.py call
                )
                # ---- MODIFICATION END ----
                samples.extend(mol_list) # samples collects all individual molecule_lists
                batch_groups_reactants.append(mol_list)
                batch_scores_reactants.append(current_scores) # current_scores from sample_batch
                if s_idx == 0:
                    ground_truth.extend(true_mol_list)

            ident += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

            # Regrouping sampled reactants for computing top-N accuracy
            for mol_idx_in_batch in range(bs):
                mol_samples_group = []
                mol_scores_group = []
                for reactant_batch_group, score_batch_group in zip(batch_groups_reactants, batch_scores_reactants):
                    if mol_idx_in_batch < len(reactant_batch_group): # Check index bounds
                        mol_samples_group.append(reactant_batch_group[mol_idx_in_batch])
                        mol_scores_group.append(score_batch_group[mol_idx_in_batch])
                    # else: print(f"Warning: mol_idx_in_batch {mol_idx_in_batch} out of bounds for reactant_batch_group len {len(reactant_batch_group)}")


                if len(mol_samples_group) == self.samples_per_input: # Ensure correct number of samples per input
                    grouped_samples.append(mol_samples_group)
                    grouped_scores.append(mol_scores_group)
                # else: print(f"Warning: Incorrect number of samples for input {ident - to_generate + mol_idx_in_batch}. Expected {self.samples_per_input}, got {len(mol_samples_group)}")


        # ---- MODIFICATION START ----
        # Ensure ground_truth and grouped_samples are not empty before proceeding
        if not ground_truth or not grouped_samples:
            print("Warning: No samples generated or no ground truth available for sampling metrics. Skipping.")
        else:
            atom_decoder_to_use = self.dataset_info.atom_decoder if hasattr(self.dataset_info, 'atom_decoder') else []
            if not atom_decoder_to_use:
                print("Warning: atom_decoder not available in dataset_info. Retrosynthesis metrics might be inaccurate.")

            # Check if grouped_scores has content and matches grouped_samples length
            scores_for_metrics = grouped_scores if len(grouped_scores) == len(grouped_samples) else None
            if scores_for_metrics is None and grouped_scores: # If lengths mismatch but scores exist
                print(f"Warning: Mismatch in length of grouped_scores ({len(grouped_scores)}) and grouped_samples ({len(grouped_samples)}). Scores will not be used for metrics.")


            to_log_retro = compute_retrosynthesis_metrics(
                grouped_samples=grouped_samples,
                ground_truth=ground_truth,
                atom_decoder=atom_decoder_to_use,
                grouped_scores=scores_for_metrics, # Pass the (potentially None) scores
            )
            for metric_name, metric in to_log_retro.items():
                self.log(f'sampling_retro/{metric_name}', metric) # Added prefix for clarity

        if not samples:
            print("Warning: No samples generated overall. Skipping general sampling metrics.")
        else:
            # ---- MODIFICATION START: Use renamed attribute ----
            if hasattr(self, 'sampling_metrics_module') and callable(self.sampling_metrics_module.forward):
            # ---- MODIFICATION END ----
                 to_log_sampling = self.sampling_metrics_module(samples) # Pass all collected samples
                 for metric_name, metric in to_log_sampling.items():
                     self.log(f'sampling_general/{metric_name}', metric) # Added prefix
            else:
                print("Warning: self.sampling_metrics_module not available or not callable.")
        # ---- MODIFICATION END ----

        # ---- MODIFICATION START: Use renamed attribute ----
        if hasattr(self, 'sampling_metrics_module') and callable(self.sampling_metrics_module.reset):
            self.sampling_metrics_module.reset()
        # ---- MODIFICATION END ----


    def apply_noise(self, X, E, y, X_T, E_T, y_T, node_mask):
        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        # ---- MODIFICATION START ----
        # Ensure X has a batch dimension for X.size(0)
        current_batch_size = X.size(0) if X.ndim > 1 else 1 # Handle single sample case (though unlikely in training)
        if current_batch_size == 0: # Should not happen
            # print("Warning: Batch size is 0 in apply_noise. Returning empty noisy_data.")
            return { # Return a structure that won't break downstream, but is empty
                't_int': torch.empty((0,1), device=X.device, dtype=torch.float), 't': torch.empty((0,1), device=X.device, dtype=torch.float),
                'beta_t': torch.empty((0,1), device=X.device, dtype=torch.float), 'alpha_s_bar': torch.empty((0,1), device=X.device, dtype=torch.float),
                'alpha_t_bar': torch.empty((0,1), device=X.device, dtype=torch.float),
                'X_t': torch.empty_like(X), 'E_t': torch.empty_like(E), 'y_t': torch.empty_like(y),
                'node_mask': torch.empty_like(node_mask)
            }

        t_int = torch.randint(lowest_t, self.T + 1, size=(current_batch_size, 1), device=X.device).float()
        # ---- MODIFICATION END ----
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar( # InterpolationTransition.get_Qt_bar
            alpha_bar_t=alpha_t_bar,
            X_T=X_T,
            E_T=E_T,
            y_T=y_T, # y_T is likely empty, but passed for completeness
            node_mask=node_mask,
            device=self.device,
        )  # Returns PlaceHolder with X:(bs, n, dx_in, dx_out), E:(bs, n, n, de_in, de_out)

        if not (len(Qtb.X.shape) == 4 and len(Qtb.E.shape) == 5): # Added check
            raise ValueError(f"Unexpected shape for Qtb.X ({Qtb.X.shape}) or Qtb.E ({Qtb.E.shape})")
        if not (abs(Qtb.X.sum(dim=3) - 1.) < 1e-4).all(): # Added check
            # print(f"Warning: Sum of Qtb.X probabilities is not 1: {Qtb.X.sum(dim=3)}")
            # Potentially normalize here if needed, though it should be correct from transition_model
            pass
        if not (abs(Qtb.E.sum(dim=4) - 1.) < 1e-4).all(): # Added check
            # print(f"Warning: Sum of Qtb.E probabilities is not 1: {Qtb.E.sum(dim=4)}")
            pass


        probX = (X.unsqueeze(-2) @ Qtb.X).squeeze(-2)  # (bs, n, dx_out)
        probE = (E.unsqueeze(-2) @ Qtb.E).squeeze(-2)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output).float()
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output).float()
        if not (X.shape == X_t.shape and E.shape == E_t.shape): # Added check
            raise ValueError(f"Shape mismatch after noise: X_orig={X.shape}, X_t={X_t.shape}; E_orig={E.shape}, E_t={E_t.shape}")


        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask) # y is original y (likely empty)

        noisy_data = {
            't_int': t_int,
            't': t_float,
            'beta_t': beta_t,
            'alpha_s_bar': alpha_s_bar,
            'alpha_t_bar': alpha_t_bar,
            'X_t': z_t.X,
            'E_t': z_t.E,
            'y_t': z_t.y, # This y_t is the original context y, not a noised version of y_T
            'node_mask': node_mask
        }
        return noisy_data

    # ---- MODIFICATION START ----
    # self.model() returns two values now. This forward method needs to adapt.
    def forward(self, noisy_data, extra_data, node_mask):
        X_in = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E_in = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y_in = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        
        # self.model is GraphTransformer instance
        final_placeholder_output, intermediate_y = self.model(X_in, E_in, y_in, node_mask)
        
        return final_placeholder_output, intermediate_y # Return both
    # ---- MODIFICATION END ----


    @torch.no_grad()
    def sample_batch( # This method is for generating samples during validation/testing
            self,
            data,
            batch_id,
            batch_size,
            keep_chain, # For visualization
            number_chain_steps_to_save, # For visualization
            save_final, # For visualization
            sample_idx, # Index of the sample per input (e.g. if n_samples=10, this is 0 to 9)
            save_true_reactants=True, # Original default
            use_one_hot=False, # Passed from sample.py
    ):

        chain_X, chain_E, true_molecule_list, products_list, molecule_list, _, nll, ell = self.sample_chain( # Added _ for pred_z0_one_hot_final
            data=data,
            batch_size=batch_size,
            keep_chain=keep_chain,
            number_chain_steps_to_save=number_chain_steps_to_save,
            save_true_reactants=save_true_reactants,
            use_one_hot=use_one_hot,
        )

        if self.visualization_tools is not None and save_final > 0 : # Added save_final > 0 condition
            self.visualize( # visualize method unchanged, relies on chain_X/E etc.
                chain_X=chain_X,
                chain_E=chain_E,
                true_molecule_list=true_molecule_list,
                products_list=products_list,
                molecule_list=molecule_list,
                sample_idx=sample_idx,
                batch_id=batch_id,
                save_final=save_final # This parameter controls how many are visualized
            )
        
        # The original sample.py expects scores, nll, ell.
        # Scores were [0]*len(molecule_list). We maintain this for now.
        # nll and ell are actual lists of floats from sample_chain.
        return molecule_list, true_molecule_list, products_list, [0] * len(molecule_list), nll, ell

    # This method is the core reverse process sampling logic
    def sample_chain(
            self, data, batch_size, keep_chain, number_chain_steps_to_save, save_true_reactants, use_one_hot=False
    ):
        # Context product (z_T in Markov Bridge formulation, which is the product molecule)
        product, node_mask = utils.to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product = product.mask(node_mask) # product.X is (bs, Np, df), product.E is (bs, Np, Np, de)

        # Creating context for the model (usually the product itself if use_context is True)
        context = product.clone() if self.use_context else None

        # Masks for fixed and modifiable nodes (based on dummy atoms in product)
        # These will have shape (bs, Np, 1)
        fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1) 
        modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
        if not torch.all(fixed_nodes | modifiable_nodes):
            # print("Warning (sample_chain): Not all product nodes are categorized as fixed or modifiable.")
            pass

        # z_T – starting state for reverse process is the product
        X, E = product.X, product.E # X is (bs, Np, df_onehot), E is (bs, Np, Np, de_onehot)
        y = torch.empty((node_mask.shape[0], 0), device=self.device, dtype=X.dtype) # Original y for z_T

        if not (E == torch.transpose(E, 1, 2)).all():
            # print("Warning (sample_chain): Initial E is not symmetric.")
            pass
        
        # Ensure number_chain_steps_to_save is valid
        current_T = self.T if hasattr(self, 'T') and self.T is not None else 500 # Fallback if T not set
        if not (number_chain_steps_to_save < current_T and number_chain_steps_to_save > 0):
            # print(f"Warning (sample_chain): number_chain_steps_to_save ({number_chain_steps_to_save}) invalid for T={current_T}. Adjusting.")
            number_chain_steps_to_save = max(1, min(number_chain_steps_to_save, current_T -1 if current_T > 1 else 1) )


        # Initialize chain storage only if keep_chain > 0
        chain_X = torch.empty(0, device=X.device, dtype=torch.long) # Default to empty long tensor for discrete X
        chain_E = torch.empty(0, device=E.device, dtype=torch.long) # Default to empty long tensor for discrete E

        if keep_chain > 0:
            # Determine Np (max nodes in this batch from product) for chain storage
            Np_for_chain = X.size(1) 
            chain_X_size = torch.Size((number_chain_steps_to_save, keep_chain, Np_for_chain))
            chain_E_size = torch.Size((number_chain_steps_to_save, keep_chain, Np_for_chain, Np_for_chain))
            chain_X = torch.zeros(chain_X_size, device=X.device, dtype=torch.long) # Store discrete (integer) class indices
            chain_E = torch.zeros(chain_E_size, device=E.device, dtype=torch.long) # Store discrete (integer) class indices


        # ---- RESTORED: nll and ell accumulators ----
        nll_accum = torch.zeros(batch_size, device=X.device, dtype=torch.float64) 
        ell_accum = torch.zeros(batch_size, device=X.device, dtype=torch.float64) 
        # ---- RESTORED END ----

        # This placeholder will hold the one-hot state z_s after each sampling step
        # It's initialized here to define its scope outside the loop for the final assignment
        sampled_s_placeholder_one_hot = utils.PlaceHolder(X=X.clone(), E=E.clone(), y=y.clone())


        for s_int_loop_val in tqdm(reversed(range(0, current_T)), total=current_T, desc="Sampling chain"): 
            s_array = s_int_loop_val * torch.ones((batch_size, 1), device=self.device, dtype=torch.float) 
            t_array = s_array + 1 
            
            s_norm = s_array / current_T 
            t_norm = t_array / current_T 

            # sampled_s_placeholder_one_hot is the one-hot version of z_s
            # discrete_version_from_sampler is the integer (collapsed) version of z_s from sampler *before* fix_product_nodes
            # ---- RESTORED: nll and ell from sample_p_zs_given_zt ----
            sampled_s_placeholder_one_hot, discrete_version_from_sampler, node_log_likelihood, edge_log_likelihood = self.sample_p_zs_given_zt(
            # ---- RESTORED END ----
                s=s_norm, 
                t=t_norm, 
                X_t=X,    # Current state (one-hot)
                E_t=E,    # Current state (one-hot)
                y_t=y,    # Current state (usually empty)
                X_T=product.X, # Anchoring state (product, one-hot)
                E_T=product.E, # Anchoring state (product, one-hot)
                y_T=product.y, # Anchoring state (product, usually empty)
                node_mask=node_mask,
                context=context, # Context for the model (usually product)
                use_one_hot=use_one_hot, # Passed from sample_batch
            )

            # This variable will hold the discrete representation for chain visualization
            discrete_sampled_s_for_chain_viz = None

            if self.fix_product_nodes:
                # Apply fix_product_nodes to the one-hot representation
                sampled_s_placeholder_one_hot.X = sampled_s_placeholder_one_hot.X * modifiable_nodes + product.X * fixed_nodes
                sampled_s_placeholder_one_hot = sampled_s_placeholder_one_hot.mask(node_mask) # Apply mask after modification
                
                # For chain visualization, get the discrete version of the *fixed* one-hot state
                if keep_chain > 0:
                    discrete_sampled_s_for_chain_viz = sampled_s_placeholder_one_hot.clone().mask(node_mask, collapse=True)
            else:
                # If not fixing nodes, the discrete version from sampler is what represents the state for the chain
                if keep_chain > 0:
                    discrete_sampled_s_for_chain_viz = discrete_version_from_sampler # This is already collapsed

            # Update current state (X, E, y) with the *one-hot, potentially fixed* z_s for the next iteration
            X, E, y = sampled_s_placeholder_one_hot.X, sampled_s_placeholder_one_hot.E, sampled_s_placeholder_one_hot.y

            # Save to chain for visualization (if enabled and discrete version is available)
            if keep_chain > 0 and discrete_sampled_s_for_chain_viz is not None:
                 write_index = int((s_int_loop_val * number_chain_steps_to_save) / current_T)
                 if 0 <= write_index < number_chain_steps_to_save:
                     # Ensure discrete_sampled_s_for_chain_viz.X has the correct shape for chain_X
                     if discrete_sampled_s_for_chain_viz.X.shape == chain_X[write_index, :keep_chain].shape:
                         chain_X[write_index, :keep_chain] = discrete_sampled_s_for_chain_viz.X[:keep_chain]
                         chain_E[write_index, :keep_chain] = discrete_sampled_s_for_chain_viz.E[:keep_chain]
                     # else:
                         # print(f"Warning: Shape mismatch for chain viz. Discrete X: {discrete_sampled_s_for_chain_viz.X.shape}, Chain slice: {chain_X[write_index, :keep_chain].shape}")
            
            # ---- RESTORED: Accumulate nll and ell ----
            nll_accum += node_log_likelihood
            ell_accum += edge_log_likelihood
            # ---- RESTORED END ----

        # After loop, sampled_s_placeholder_one_hot holds the final z_0 (one-hot, potentially fixed)
        pred_z0_one_hot_final = sampled_s_placeholder_one_hot.clone() # This is the model's prediction for z0 (reactant)

        # Collapse the final, potentially fixed, z_0 to get discrete representation for output molecules
        sampled_z0_discrete_final = sampled_s_placeholder_one_hot.mask(node_mask, collapse=True)
        X_final_discrete, E_final_discrete = sampled_z0_discrete_final.X, sampled_z0_discrete_final.E

        if keep_chain > 0 and chain_X.numel() > 0: # Ensure chain was initialized and has elements
            # The chain was filled such that reversing it shows t=1 to t=T (approx)
            # The last state computed (X_final_discrete, E_final_discrete) is z_0.
            # We want the frame at s_int_loop_val=0 (which means t_norm=1/T, model_perspective_t close to 1)
            # to contain z_0. The write_index for s_int_loop_val=0 is 0.
            if number_chain_steps_to_save > 0: # Check valid index for chain_X[0]
                 chain_X[0, :keep_chain] = X_final_discrete[:keep_chain]
                 chain_E[0, :keep_chain] = E_final_discrete[:keep_chain]

            chain_X = diffusion_utils.reverse_tensor(chain_X) 
            chain_E = diffusion_utils.reverse_tensor(chain_E) 

            # Repeat the state representing z_0 (which is now the last frame after reversing)
            chain_X = torch.cat([chain_X, chain_X[-1:, ...].repeat(10, 1, 1) if chain_X.numel() > 0 else chain_X], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:, ...].repeat(10, 1, 1, 1) if chain_E.numel() > 0 else chain_E], dim=0)
            if chain_X.numel() > 0 and not (chain_X.shape[0] == (number_chain_steps_to_save + 10)):
                 pass # print(f"Warning (sample_chain): Final chain_X length check failed.")


        true_molecule_list_out = utils.create_true_reactant_molecules(data, batch_size) if save_true_reactants else []
        products_list_out = utils.create_input_product_molecules(data, batch_size)
        molecule_list_out = utils.create_pred_reactant_molecules(X_final_discrete, E_final_discrete, data.batch, batch_size)

        return (
            chain_X, chain_E, 
            true_molecule_list_out, products_list_out, molecule_list_out, 
            pred_z0_one_hot_final, # Return the final one-hot prediction for z0
            # ---- RESTORED: Return nll and ell lists ----
            nll_accum.detach().cpu().numpy().tolist(),
            ell_accum.detach().cpu().numpy().tolist(),
            # ---- RESTORED END ----
        )

    def visualize( # This method seems fine, just ensure parameters are passed correctly
            self,
            chain_X, chain_E,
            true_molecule_list, products_list, molecule_list,
            sample_idx, batch_id, save_final
    ):
        # Ensure visualization_tools is not None
        if self.visualization_tools is None:
            return

        # ---- MODIFICATION START: Use self.current_epoch + 1 for display ----
        epoch_display = self.current_epoch + 1 if hasattr(self, 'current_epoch') else 'unknown_epoch'
        current_samples_path = os.path.join(self.graphs_dir, f'epoch{epoch_display}_b{batch_id}')
        current_chains_dir = os.path.join(self.chains_dir, f'epoch_{epoch_display}')
        # ---- MODIFICATION END ----
        os.makedirs(current_samples_path, exist_ok=True) # Added exist_ok
        os.makedirs(current_chains_dir, exist_ok=True)   # Added exist_ok

        if sample_idx == 0: # Only visualize chains and true/input for the first sample of a batch
            # 1. Visualize chains
            if chain_X.numel() > 0 and chain_E.numel() > 0 : # Check if chains are not empty
                num_molecules_in_chain = chain_X.shape[1] # Number of molecules for which chain was saved (keep_chain)
                for i in range(num_molecules_in_chain):
                    results_path_chain = os.path.join(current_chains_dir, f'molecule_{batch_id + i}') # Renamed for clarity
                    os.makedirs(results_path_chain, exist_ok=True)
                    self.visualization_tools.visualize_chain(
                        path=results_path_chain,
                        nodes_list=chain_X[:, i, :].cpu().numpy(), # Added .cpu()
                        adjacency_matrix=chain_E[:, i, :].cpu().numpy(), # Added .cpu()
                    )

            # 2. Visualize true reactants
            if true_molecule_list: # Check not empty
                self.visualization_tools.visualize(
                    path=current_samples_path,
                    molecules=true_molecule_list,
                    num_molecules_to_visualize=save_final,
                    prefix='true_',
                )

            # 3. Visualize input products
            if products_list: # Check not empty
                self.visualization_tools.visualize(
                    path=current_samples_path,
                    molecules=products_list,
                    num_molecules_to_visualize=save_final,
                    prefix='input_product_',
                )

        # Visualize predicted reactants for every sample_idx
        if molecule_list: # Check not empty
            self.visualization_tools.visualize(
                path=current_samples_path,
                molecules=molecule_list,
                num_molecules_to_visualize=save_final,
                prefix=f'pred_',
                suffix=f'_{sample_idx}'
            )

    # sample_p_zs_given_zt is the one-step reverse conditional probability for Markov Bridge
    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, X_T, E_T, y_T, node_mask, context=None, use_one_hot=False):
        # s: normalized time for z_s (target state, t-1)
        # t: normalized time for z_t (current state, t)
        # X_t, E_t, y_t: current state features (one-hot)
        # X_T, E_T, y_T: anchoring state features (product molecule, z_T, one-hot)
        
        bs, n, _ = X_t.shape # Added _ to unpack, was bs, n = X_t.shape[:2]
        
        # "Hack": in direct MB we consider flipped time flow for the model's perspective on beta_t
        # The model is trained to predict z_0 given z_t and z_T.
        # When sampling z_{s} from z_{t}, the model effectively sees (t_model = 1 - t_physical)
        # where t_physical is the time from z_T (product).
        # So, if current physical time t is close to T (i.e., t_norm close to 1), model sees time close to 0.
        model_perspective_t = 1 - t # t is t_norm from sample_chain
        beta_t_for_model = self.noise_schedule(t_normalized=model_perspective_t)  # (bs, 1)

        # Neural net predictions for z_0 based on current z_t (at model_perspective_t) and z_T (context)
        # noisy_data dict for model input: X_t, E_t are current state, 't' is model_perspective_t
        noisy_data_for_model = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 
                                't': model_perspective_t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data_for_model, context=context)
        
        # ---- MODIFICATION START ----
        # self.forward now returns (pred_placeholder, intermediate_y)
        # We only need pred_placeholder here for the retro prediction.
        pred_z0_placeholder, _ = self.forward(noisy_data_for_model, extra_data, node_mask)
        # ---- MODIFICATION END ----

        # Normalize predictions for z_0
        pred_X0_probs = F.softmax(pred_z0_placeholder.X, dim=-1)  # bs, n, d0
        pred_E0_probs = F.softmax(pred_z0_placeholder.E, dim=-1)  # bs, n, n, d0

        if use_one_hot: # If true, convert soft predictions to one-hot based on argmax
            x_mask_internal = node_mask.unsqueeze(-1).float() # Renamed to avoid conflict
            e_mask1_internal = x_mask_internal.unsqueeze(2).float()
            e_mask2_internal = x_mask_internal.unsqueeze(1).float()
            pred_X0_probs = F.one_hot(pred_z0_placeholder.X.argmax(dim=-1), num_classes=self.Xdim_output).float() * x_mask_internal
            pred_E0_probs = F.one_hot(pred_z0_placeholder.E.argmax(dim=-1), num_classes=self.Edim_output).float() * e_mask1_internal * e_mask2_internal


        # Compute transition matrices q(z_s | z_0, z_t)
        # This requires q(z_s | z_0) and q(z_t | z_0) and then combining them.
        # More directly, for discrete Markov Bridge (Hoogeboom et al. 2021, Eq. 10 for q(x_{t-1}|x_t, x_0)):
        # q(z_s | z_t, z_0_pred, z_T_anchor) \propto sum_{z'_0} p(z_s | z'_0, z_T_anchor) p(z_t | z'_0, z_T_anchor) p_theta(z'_0 | z_t, z_T_anchor)
        # This is complex. The original code used simpler InterpolationTransition.get_Qt for p(z_s | z_t, z_T_pred_as_anchor)
        # Let's re-verify the transition logic of InterpolationTransition
        # InterpolationTransition.get_Qt(beta_t, X_T, E_T, ...) defines q(z_s | z_t) where beta_t is for one step,
        # and X_T, E_T are the *target* state for interpolation (which here is pred_X0_probs, pred_E0_probs).
        # The `beta_t` in `get_Qt` determines how much z_t transitions towards X_T.
        # For a single step s from t, beta_t should be related to the noise added in one step.
        # The `beta_t_for_model` used above was for the model's prediction of z0.
        # For the actual transition from z_t to z_s, we need a beta corresponding to this single step.
        # Let's assume beta_t_for_model is the correct one-step noise rate parameter.
        
        Qt_one_step = self.transition_model.get_Qt( # InterpolationTransition.get_Qt
            beta_t=beta_t_for_model, # This beta should control the step from z_t to z_s using pred_X0/E0 as target
            X_T=pred_X0_probs, # The model's prediction of z0 acts as the target for interpolation
            E_T=pred_E0_probs,
            y_T=y_T, # y_T is product's y (likely empty), not used by current InterpolationTransition for X, E
            node_mask=node_mask,
            device=self.device,
        )  # Returns PlaceHolder with X:(bs, n, dx_in, dx_out), E:(bs, n, n, de_in, de_out)

        # Node transition probabilities for z_s given z_t and model's prediction of z_0
        # P(z_s | z_t) = z_t @ Q_onestep.X
        unnormalized_prob_X_s = X_t.unsqueeze(-2) @ Qt_one_step.X  # bs, n, 1, d_out(s)
        unnormalized_prob_X_s = unnormalized_prob_X_s.squeeze(-2)  # bs, n, d_out(s)
        # Handle cases where sum is zero to avoid NaN, by setting to uniform if that happens.
        sum_probs_X = torch.sum(unnormalized_prob_X_s, dim=-1, keepdim=True)
        unnormalized_prob_X_s[sum_probs_X.squeeze(-1) == 0] = 1.0 / self.Xdim_output # Uniform if sum is zero
        # Renormalize just in case, though get_Qt should return stochastic matrices.
        prob_X_s = unnormalized_prob_X_s / torch.sum(unnormalized_prob_X_s, dim=-1, keepdim=True)

        # Edge transition probabilities
        E_t_flat = E_t.flatten(start_dim=1, end_dim=2)  # (bs, N, d_in(t))
        Qt_E_flat = Qt_one_step.E.flatten(start_dim=1, end_dim=2)  # (bs, N, d_in(t), d_out(s))
        unnormalized_prob_E_s = E_t_flat.unsqueeze(-2) @ Qt_E_flat  # bs, N, 1, d_out(s)
        unnormalized_prob_E_s = unnormalized_prob_E_s.squeeze(-2)  # bs, N, d_out(s)
        sum_probs_E = torch.sum(unnormalized_prob_E_s, dim=-1, keepdim=True)
        unnormalized_prob_E_s[sum_probs_E.squeeze(-1) == 0] = 1.0 / self.Edim_output # Uniform if sum is zero
        prob_E_s = unnormalized_prob_E_s / torch.sum(unnormalized_prob_E_s, dim=-1, keepdim=True)
        prob_E_s = prob_E_s.reshape(bs, n, n, pred_E0_probs.shape[-1]) # Reshape back

        if not ((prob_X_s.sum(dim=-1) - 1).abs() < 1e-3).all(): # Relaxed tolerance
            # print(f"Warning (sample_p_zs): prob_X_s does not sum to 1. Max diff: {((prob_X_s.sum(dim=-1) - 1).abs()).max()}")
            pass
        if not ((prob_E_s.sum(dim=-1) - 1).abs() < 1e-3).all(): # Relaxed tolerance
            # print(f"Warning (sample_p_zs): prob_E_s does not sum to 1. Max diff: {((prob_E_s.sum(dim=-1) - 1).abs()).max()}")
            pass

        sampled_s = diffusion_utils.sample_discrete_features(prob_X_s, prob_E_s, node_mask=node_mask)

        X_s_one_hot = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s_one_hot = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        if not (E_s_one_hot == torch.transpose(E_s_one_hot, 1, 2)).all(): # Added check
            # print("Warning (sample_p_zs): Sampled E_s_one_hot is not symmetric.")
            # Symmetrize if necessary, though sample_discrete_features should handle it.
            pass
        if not (X_t.shape == X_s_one_hot.shape and E_t.shape == E_s_one_hot.shape): # Added check
            raise ValueError(f"Shape mismatch in sample_p_zs: X_t={X_t.shape}, X_s={X_s_one_hot.shape}; E_t={E_t.shape}, E_s={E_s_one_hot.shape}")


        out_one_hot_placeholder = utils.PlaceHolder(X=X_s_one_hot, E=E_s_one_hot, y=torch.zeros_like(y_t)) # y for z_s, usually empty
        # For discrete output, we use the same X_s_one_hot, E_s_one_hot and collapse them later.
        # The original code had out_discrete = utils.PlaceHolder(X=X_s, E=E_s, ...) using integer X_s, E_s.
        # For consistency with how mask(collapse=True) works, let's keep it as one-hot for now.
        # The .mask(collapse=True) will handle the conversion to integer indices.
        out_discrete_placeholder = utils.PlaceHolder(X=X_s_one_hot.clone(), E=E_s_one_hot.clone(), y=torch.zeros_like(y_t))


        # ---- RESTORED: Likelihood calculation for VLB ----
        # Likelihood for VLB (log p_theta(z_s | z_t, z_T_anchor))
        # This is essentially log prob_X_s[sampled_s.X] + log pred_X0_probs[sampled_s.X_from_pred_model?]
        # The NLL/ELL calculation from the original sample_chain was:
        # node_log_likelihood = torch.log(prob_X) + torch.log(pred_X)
        # node_log_likelihood = (node_log_likelihood * X_s).sum(-1).sum(-1)
        # Here, prob_X is prob_X_s, and pred_X is pred_X0_probs. X_s is X_s_one_hot.
        
        # Ensure probabilities are not zero before log
        safe_prob_X_s = torch.clamp(prob_X_s, min=1e-9)
        # safe_pred_X0_probs = torch.clamp(pred_X0_probs, min=1e-9) # Not directly used here for this formulation of log p(z_s|z_t)
        log_prob_transition_X = torch.log(safe_prob_X_s) # log q(z_s | z_t, z_0_pred_from_theta)
        
        # Contribution to ELBO from this step (using sampled z_s as true for next step)
        # This is part of log p_theta(z_s | z_t)
        # For a specific z_s sampled, this is log (sum_{z_0} q(z_s|z_t,z_0) p_theta(z_0|z_t) )
        # The sum over z_0 is implicitly done by using pred_X0_probs in get_Qt
        # So, node_log_likelihood_step is log q(z_s_sampled | z_t, z_0_from_theta_dist)
        node_ll_step = (log_prob_transition_X * X_s_one_hot).sum(dim=-1) # Sum over classes
        node_ll_step = node_ll_step.sum(dim=-1) # Sum over nodes in graph

        safe_prob_E_s = torch.clamp(prob_E_s, min=1e-9)
        # safe_pred_E0_probs = torch.clamp(pred_E0_probs, min=1e-9) # Not needed for this formulation
        log_prob_transition_E = torch.log(safe_prob_E_s)
        
        edge_ll_step = (log_prob_transition_E * E_s_one_hot).sum(dim=-1) # Sum over classes
        # Sum over edges (upper triangle to avoid double counting)
        # Create a mask for the upper triangle
        upper_triangle_mask = torch.triu(torch.ones_like(edge_ll_step), diagonal=1).bool()
        # Original was: edge_ll_step = edge_ll_step[upper_triangle_mask].sum() / bs # Average per graph in batch
        # To match node_ll_step shape (bs), sum per graph:
        edge_ll_step_sum_per_graph = torch.zeros(bs, device=edge_ll_step.device, dtype=edge_ll_step.dtype)
        for i in range(bs):
            edge_ll_step_sum_per_graph[i] = edge_ll_step[i][upper_triangle_mask[i]].sum()
        edge_ll_step = edge_ll_step_sum_per_graph
        # ---- RESTORED END ----


        return (
            out_one_hot_placeholder.mask(node_mask).type_as(y_t),
            out_discrete_placeholder.mask(node_mask, collapse=True).type_as(y_t), # This gives integer indices
            # ---- RESTORED: Return nll and ell ----
            node_ll_step, # Per batch item
            edge_ll_step, # Per batch item
            # ---- RESTORED END ----
        )
    
    # VLB helper: q(z_{s} | z_t, z_0_true) where z_0_true is actual reactant
    def compute_q_zs_given_q_zt(self, z_t, z_0_true, node_mask, t): # t is model_perspective_t here
        X_t_one_hot = z_t.X.to(torch.float32) # Current z_t (one-hot)
        E_t_one_hot = z_t.E.to(torch.float32)

        beta_t_for_model = self.noise_schedule(t_normalized=t)  # (bs, 1)

        # True z_0 (reactants)
        X_0_true_one_hot = z_0_true.X.to(torch.float32)
        E_0_true_one_hot = z_0_true.E.to(torch.float32)
        y_0_true = z_0_true.y # Likely empty

        # Transition from z_t towards z_0_true
        Qt_one_step_true = self.transition_model.get_Qt(
            beta_t=beta_t_for_model,
            X_T=X_0_true_one_hot, # Target for interpolation is true z_0
            E_T=E_0_true_one_hot,
            y_T=y_0_true,
            node_mask=node_mask,
            device=self.device,
        )

        # Probabilities q(z_s | z_t, z_0_true)
        prob_X_s_true = (X_t_one_hot.unsqueeze(-2) @ Qt_one_step_true.X).squeeze(-2)
        prob_X_s_true = prob_X_s_true / torch.sum(prob_X_s_true, dim=-1, keepdim=True) # Ensure normalization

        bs_internal, n_internal, _ = X_t_one_hot.shape # Renamed internal vars
        E_t_flat_internal = E_t_one_hot.flatten(start_dim=1, end_dim=2)
        Qt_E_flat_internal = Qt_one_step_true.E.flatten(start_dim=1, end_dim=2)
        prob_E_s_true = (E_t_flat_internal.unsqueeze(-2) @ Qt_E_flat_internal).squeeze(-2)
        prob_E_s_true = prob_E_s_true / torch.sum(prob_E_s_true, dim=-1, keepdim=True)
        prob_E_s_true = prob_E_s_true.reshape(bs_internal, n_internal, n_internal, E_0_true_one_hot.shape[-1])
        
        # Added assertions for sum to 1, with tolerance
        if not ((prob_X_s_true.sum(dim=-1) - 1).abs() < 1e-3).all():
            # print(f"Warning (compute_q_zs): prob_X_s_true does not sum to 1. Max diff: {((prob_X_s_true.sum(dim=-1) - 1).abs()).max()}")
            pass
        if not ((prob_E_s_true.sum(dim=-1) - 1).abs() < 1e-3).all():
            # print(f"Warning (compute_q_zs): prob_E_s_true does not sum to 1. Max diff: {((prob_E_s_true.sum(dim=-1) - 1).abs()).max()}")
            pass

        return prob_X_s_true, prob_E_s_true

    # VLB helper: p_theta(z_{s} | z_t) = sum_{z_0_pred} q(z_s | z_t, z_0_pred) p_theta(z_0_pred | z_t)
    def compute_p_zs_given_p_zt(self, z_t, pred_z0_from_model, node_mask, t): # t is model_perspective_t
        # pred_z0_from_model is the PlaceHolder output of the main model (predicting z0)
        
        # Softmax probabilities for the model's prediction of z_0
        p_X0_theta_probs = F.softmax(pred_z0_from_model.X, dim=-1)  # bs, n, d0
        p_E0_theta_probs = F.softmax(pred_z0_from_model.E, dim=-1)  # bs, n, n, d0
        # p_y0_theta_probs = F.softmax(pred_z0_from_model.y, dim=-1) # if y is predicted

        # Initialize P(z_s | z_t) by summing over all possible z_0 predictions, weighted by p_theta(z_0|z_t)
        # This is computationally expensive if done by explicit sum.
        # Instead, we use the fact that pred_X0_probs IS p_theta(z_0|z_t) effectively.
        # So, q(z_s | z_t, z_0_pred_from_theta_dist) using p_X0_theta_probs as X_T in get_Qt
        
        X_t_one_hot = z_t.X.to(torch.float32)
        E_t_one_hot = z_t.E.to(torch.float32)
        y_t_one_hot = z_t.y.to(torch.float32) # Though likely empty

        beta_t_for_model = self.noise_schedule(t_normalized=t)

        Qt_one_step_pred = self.transition_model.get_Qt(
            beta_t=beta_t_for_model,
            X_T=p_X0_theta_probs, # Using the model's predicted distribution for z0 as the target
            E_T=p_E0_theta_probs,
            y_T=y_t_one_hot, # Should be y0_pred if available, y_t for structure
            node_mask=node_mask,
            device=self.device,
        )

        prob_X_s_pred = (X_t_one_hot.unsqueeze(-2) @ Qt_one_step_pred.X).squeeze(-2)
        prob_X_s_pred = prob_X_s_pred / torch.sum(prob_X_s_pred, dim=-1, keepdim=True)

        bs_internal, n_internal, _ = X_t_one_hot.shape
        E_t_flat_internal = E_t_one_hot.flatten(start_dim=1, end_dim=2)
        Qt_E_flat_internal = Qt_one_step_pred.E.flatten(start_dim=1, end_dim=2)
        prob_E_s_pred = (E_t_flat_internal.unsqueeze(-2) @ Qt_E_flat_internal).squeeze(-2)
        prob_E_s_pred = prob_E_s_pred / torch.sum(prob_E_s_pred, dim=-1, keepdim=True)
        prob_E_s_pred = prob_E_s_pred.reshape(bs_internal, n_internal, n_internal, p_E0_theta_probs.shape[-1])
        
        if not ((prob_X_s_pred.sum(dim=-1) - 1).abs() < 1e-3).all():
            # print(f"Warning (compute_p_zs): prob_X_s_pred does not sum to 1. Max diff: {((prob_X_s_pred.sum(dim=-1) - 1).abs()).max()}")
            pass
        if not ((prob_E_s_pred.sum(dim=-1) - 1).abs() < 1e-3).all():
            # print(f"Warning (compute_p_zs): prob_E_s_pred does not sum to 1. Max diff: {((prob_E_s_pred.sum(dim=-1) - 1).abs()).max()}")
            pass
            
        return prob_X_s_pred, prob_E_s_pred


    def compute_extra_data(self, noisy_data, context=None, condition_on_t=True):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        # ---- MODIFICATION: Use correct attribute names ----
        if not hasattr(self, 'extra_features_module'):
            raise AttributeError("MarkovBridge object is missing 'extra_features_module'. Check __init__.")
        if not hasattr(self, 'domain_features_module'):
            raise AttributeError("MarkovBridge object is missing 'domain_features_module'. Check __init__.")

        # extra_features was the name of the __init__ argument,
        # self.extra_features_module is where we stored the actual module instance.
        extra_features_output = self.extra_features_module(noisy_data)
        extra_molecular_features_output = self.domain_features_module(noisy_data)
        # ---- END MODIFICATION ----

        # Renamed local variables to avoid confusion if extra_features_module was named extra_features
        extra_X_calc = torch.cat((extra_features_output.X, extra_molecular_features_output.X), dim=-1)
        extra_E_calc = torch.cat((extra_features_output.E, extra_molecular_features_output.E), dim=-1)
        extra_y_calc = torch.cat((extra_features_output.y, extra_molecular_features_output.y), dim=-1)

        if context is not None:
            extra_X_calc = torch.cat((extra_X_calc, context.X), dim=-1)
            extra_E_calc = torch.cat((extra_E_calc, context.E), dim=-1)
            # Original code didn't add context.y here.
            # if hasattr(context, 'y') and context.y.numel() > 0 and context.y.size(1) > 0:
            #    extra_y_calc = torch.cat((extra_y_calc, context.y), dim=-1)


        if condition_on_t:
            # Ensure noisy_data['t'] is correctly shaped for hstack (e.g., (batch_size, 1))
            time_tensor = noisy_data['t']
            if time_tensor.ndim == 1: # If it's (batch_size), unsqueeze to (batch_size, 1)
                time_tensor = time_tensor.unsqueeze(1)
            
            # Ensure extra_y_calc has batch dimension even if initially empty from features
            if extra_y_calc.ndim == 1 and extra_y_calc.size(0) != time_tensor.size(0) : # If extra_y_calc became 1D and not batch-aligned
                 if extra_y_calc.size(0) == 0 : # if it's an empty tensor [0]
                      extra_y_calc = torch.empty(time_tensor.size(0), 0, device=time_tensor.device, dtype=time_tensor.dtype)
                 # else: handle other size mismatches if necessary

            if extra_y_calc.size(0) != time_tensor.size(0) and extra_y_calc.numel() != 0 :
                 # This indicates a more serious batch size mismatch if extra_y_calc is not empty
                 # print(f"Warning: Batch size mismatch in compute_extra_data between extra_y_calc ({extra_y_calc.shape}) and time_tensor ({time_tensor.shape})")
                 # Fallback: create empty extra_y_calc aligned with time_tensor batch size
                 extra_y_calc = torch.empty(time_tensor.size(0), 0, device=time_tensor.device, dtype=time_tensor.dtype)


            # Ensure extra_y_calc is 2D for hstack: (batch_size, num_features)
            if extra_y_calc.ndim == 1 and extra_y_calc.numel() > 0 : # If it's 1D but not empty (e.g. from a single feature)
                extra_y_calc = extra_y_calc.unsqueeze(0).expand(time_tensor.size(0), -1) # Expand to batch if it was a single row feature vector
            elif extra_y_calc.ndim == 0 : # Scalar, make it (bs, 1)
                 extra_y_calc = extra_y_calc.unsqueeze(0).unsqueeze(0).expand(time_tensor.size(0), 1)


            if extra_y_calc.size(0) == time_tensor.size(0): # Only hstack if batch sizes match
                extra_y_calc = torch.hstack((extra_y_calc, time_tensor))
            else: # If still mismatched after attempts, just use time tensor or error
                # print("Error: Could not align batch sizes for hstack in compute_extra_data. Using only time.")
                extra_y_calc = time_tensor # Fallback to just time if hstack is problematic

        return utils.PlaceHolder(X=extra_X_calc, E=extra_E_calc, y=extra_y_calc)