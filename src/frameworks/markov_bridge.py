import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os

from src.data import utils
from src.frameworks.noise_schedule import InterpolationTransition, PredefinedNoiseScheduleDiscrete
from src.frameworks import diffusion_utils
from src.metrics.train_metrics import TrainLossDiscrete, TrainLossVLB
from src.metrics.sampling_metrics import compute_retrosynthesis_metrics
from src.models.transformer_model import GraphTransformer

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from pdb import set_trace
from torchmetrics import Accuracy


class MarkovBridge(pl.LightningModule):
    def __init__(
            self,
            experiment_name,
            chains_dir,
            graphs_dir,
            checkpoints_dir,
            diffusion_steps,
            diffusion_noise_schedule,
            transition,
            lr,
            weight_decay,
            n_layers,
            hidden_mlp_dims,
            hidden_dims,
            lambda_train,
            dataset_infos,
            train_metrics,
            sampling_metrics,
            visualization_tools,
            extra_features,
            domain_features,
            use_context,
            log_every_steps,
            sample_every_val,
            samples_to_generate,
            samples_to_save,
            samples_per_input,
            chains_to_save,
            number_chain_steps_to_save,
            fix_product_nodes=False,
            loss_type='cross_entropy',
    ):

        super().__init__()

        assert loss_type in ['cross_entropy', 'vlb']

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.name = experiment_name
        self.chains_dir = chains_dir
        self.graphs_dir = graphs_dir
        self.checkpoints_dir = checkpoints_dir

        self.model_dtype = torch.float32
        self.T = diffusion_steps
        self.transition = transition

        self.lr = lr
        self.weight_decay = weight_decay

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos
        self.train_metrics = train_metrics
        self.train_loss = TrainLossDiscrete(
            lambda_train) if loss_type != 'vlb' else TrainLossVLB(lambda_train)
        self.val_loss = TrainLossDiscrete(
            lambda_train) if loss_type != 'vlb' else TrainLossVLB(lambda_train)
        self.sampling_metrics = sampling_metrics

        # <--- 新增开始 --->
        # 初始化一个用于计算多类别分类准确率的度量
        self.val_class_accuracy = Accuracy(
            task="multiclass", num_classes=self.ydim_output)
        # <--- 新增结束 --->

        self.visualization_tools = None
        self.extra_features = extra_features
        self.domain_features = domain_features
        self.use_context = use_context

        self.model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU()
        )
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            noise_schedule=diffusion_noise_schedule,
            timesteps=diffusion_steps,
        )
        self.transition_model = InterpolationTransition(
            x_classes=self.Xdim_output,
            e_classes=self.Edim_output,
            y_classes=self.ydim_output
        )

        self.save_hyperparameters(
            'experiment_name',
            'diffusion_steps',
            'diffusion_noise_schedule',
            'transition',
            'lr',
            'weight_decay',
            'n_layers',
            'hidden_mlp_dims',
            'hidden_dims',
            'lambda_train',
            'use_context',
            'fix_product_nodes',
            'loss_type'
        )

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None

        self.number_chain_steps_to_save = number_chain_steps_to_save
        self.log_every_steps = log_every_steps
        self.sample_every_val = sample_every_val
        self.samples_to_generate = samples_to_generate
        self.samples_to_save = samples_to_save
        self.samples_per_input = samples_per_input
        self.chains_to_save = chains_to_save
        self.val_counter = 0

        self.fix_product_nodes = fix_product_nodes
        self.loss_type = loss_type

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=True,
        )

    def on_train_epoch_start(self):
        self.train_loss.reset()
        self.train_metrics.reset()

    def process_and_forward(self, data):
        # 1. 准备终点 z_T = (G_R, r_true)
        #    data.y 是 r_true (真实类别), 在上一步的 to_dense 修改后，它会被正确加载
        reactants, r_node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch, y=data.y)
        reactants = reactants.mask(r_node_mask)

        # 2. 准备起点 z_0 = (G_P, r_rand)
        bs, device = data.y.size(0), data.y.device
        num_classes = data.y.size(1)  # 应该是 10

        # 为 product 的 y 从均匀分布中采样一个随机类别
        random_indices = torch.randint(0, num_classes, (bs,), device=device)
        product_y_rand = F.one_hot(
            random_indices, num_classes=num_classes).float()

        product, p_node_mask = utils.to_dense(
            data.p_x, data.p_edge_index, data.p_edge_attr, data.batch, y=product_y_rand)
        product = product.mask(p_node_mask)

        assert torch.allclose(r_node_mask, p_node_mask)
        node_mask = r_node_mask

        # 3. 调用 apply_noise 来插值
        #    X=product.X, y=product.y (随机)
        #    X_T=reactants.X, y_T=reactants.y (真实)
        noisy_data = self.apply_noise(
            X=product.X, E=product.E, y=product.y,
            X_T=reactants.X, E_T=reactants.E, y_T=reactants.y,
            node_mask=node_mask,
        )

        # 4. 后续步骤 (计算额外特征、送入模型等)
        # context 应该是已知的产物图谱，所以使用 product.clone() 是正确的
        context = product.clone() if self.use_context else None

        # !!! 关键 !!! compute_extra_data 的输入 noisy_data['y_t'] 是插值后的类别
        # 而不是真实类别或随机类别。
        extra_data = self.compute_extra_data(noisy_data, context=context)

        pred = self.forward(noisy_data, extra_data, node_mask)

        # Masking unchanged part
        if self.fix_product_nodes:
            fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
            modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
            assert torch.all(fixed_nodes | modifiable_nodes)
            pred.X = pred.X * modifiable_nodes + product.X * fixed_nodes
            pred.X = pred.X * node_mask.unsqueeze(-1)

        return reactants, product, pred, node_mask, noisy_data, context

    def training_step(self, data, i):
        reactants, product, pred, node_mask, noisy_data, _ = self.process_and_forward(
            data)
        if self.loss_type == 'vlb':
            return self.compute_training_VLB(
                reactants=reactants,
                pred=pred,
                node_mask=node_mask,
                noisy_data=noisy_data,
                i=i,
            )
        else:
            return self.compute_training_CE_loss_and_metrics(reactants=reactants, pred=pred, i=i)

    def compute_training_CE_loss_and_metrics(self, reactants, pred, i):
        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=reactants.X,
            true_E=reactants.E,
            true_y=reactants.y,
        )
        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=reactants.X,
            true_E=reactants.E,
        )

        if i % self.log_every_steps == 0:
            self.log(f'train_loss/batch_CE', loss.detach())
            for metric_name, metric in self.train_loss.compute_metrics().items():
                self.log(f'train_loss/{metric_name}', metric)
            for metric_name, metric in self.train_metrics.compute_metrics().items():
                self.log(f'train_detailed/{metric_name}/train', metric)

            self.train_loss.reset()
            self.train_metrics.reset()

        return {'loss': loss}

    def compute_validation_CE_loss(self, reactants, pred, i):
        loss = self.val_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=reactants.X,
            true_E=reactants.E,
            true_y=reactants.y,
        )

        if i % self.log_every_steps == 0:
            self.log(f'val_loss/batch_CE', loss.detach())
            for metric_name, metric in self.val_loss.compute_metrics().items():
                self.log(f'val_loss/{metric_name}', metric)

            self.train_loss.reset()
            self.train_metrics.reset()

        return {'loss': loss}

    def compute_training_VLB(self, reactants, pred, node_mask, noisy_data, i):
        z_t = utils.PlaceHolder(
            X=noisy_data['X_t'], E=noisy_data['E_t'], y=noisy_data['y_t'])
        z_T_true = reactants
        z_T_pred = pred
        t = noisy_data['t']

        true_pX, true_pE = self.compute_q_zs_given_q_zt(
            z_t, z_T_true, node_mask, t=t)
        pred_pX, pred_pE = self.compute_p_zs_given_p_zt(
            z_t, z_T_pred, node_mask, t=t)

        loss = self.train_loss(
            masked_pred_X=pred_pX,
            masked_pred_E=pred_pE,
            true_X=true_pX,
            true_E=true_pE,
        )
        if i % self.log_every_steps == 0:
            self.log(f'train_loss/batch_CE', loss.detach())
            for metric_name, metric in self.train_loss.compute_metrics().items():
                self.log(f'train_loss/{metric_name}', metric)

            self.train_loss.reset()

        return {'loss': loss}

    def compute_validation_VLB(self, reactants, pred, node_mask, noisy_data, i):
        z_t = utils.PlaceHolder(
            X=noisy_data['X_t'], E=noisy_data['E_t'], y=noisy_data['y_t'])
        z_T_true = reactants
        z_T_pred = pred
        t = noisy_data['t']

        true_pX, true_pE = self.compute_q_zs_given_q_zt(
            z_t, z_T_true, node_mask, t=t)
        pred_pX, pred_pE = self.compute_p_zs_given_p_zt(
            z_t, z_T_pred, node_mask, t=t)

        loss = self.val_loss(
            masked_pred_X=pred_pX,
            masked_pred_E=pred_pE,
            true_X=true_pX,
            true_E=true_pE,
        )
        if i % self.log_every_steps == 0:
            self.log(f'val_loss/batch_CE', loss.detach())
            for metric_name, metric in self.train_loss.compute_metrics().items():
                self.log(f'val_loss/{metric_name}', metric)

            self.train_loss.reset()

        return {'loss': loss}

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        self.sampling_metrics.reset()
        # <--- 新增开始 --->
        self.val_class_accuracy.reset()
        # <--- 新增结束 --->

    def validation_step(self, data, i):
        reactants, product, pred, node_mask, noisy_data, context = self.process_and_forward(
            data)

        # 注意：此处不再有任何关于 val_class_accuracy 的计算
        # 旧的单步准确率计算逻辑已被完全移除

        if self.loss_type == 'vlb':
            return self.compute_validation_VLB(
                reactants=reactants,
                pred=pred,
                node_mask=node_mask,
                noisy_data=noisy_data,
                i=i,
            )
        else:
            return self.compute_validation_CE_loss(reactants=reactants, pred=pred, i=i)

    def on_validation_epoch_end(self):
        # <--- 新增开始 --->
        # 计算整个验证集的最终准确率
        class_accuracy = self.val_class_accuracy.compute()

        # 打印到控制台
        print(f"\nValidation Reaction Type Accuracy: {class_accuracy:.4f}")

        # 记录到 logger (例如 wandb 或 tensorboard)
        self.log('val/class_accuracy', class_accuracy, prog_bar=True)
        # <--- 新增结束 --->

        self.val_counter += 1
        if self.val_counter % self.sample_every_val == 0:
            self.sample()
            self.trainer.save_checkpoint(
                os.path.join(self.checkpoints_dir, 'last.ckpt'))

    def sample(self):
        samples_left_to_generate = self.samples_to_generate
        samples_left_to_save = self.samples_to_save
        chains_left_to_save = self.chains_to_save

        samples = []
        grouped_samples = []
        grouped_scores = []
        ground_truth = []

        # --- 新增：初始化列表以收集类别标签 ---
        all_true_classes = []
        all_pred_classes = []
        # --- 结束新增 ---

        ident = 0
        print(f'Sampling epoch={self.current_epoch}')

        dataloader = self.trainer.datamodule.val_dataloader()
        for data in tqdm(dataloader, total=samples_left_to_generate // dataloader.batch_size):
            if samples_left_to_generate <= 0:
                break

            data = data.to(self.device)
            bs = len(data.batch.unique())
            to_generate = bs
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            batch_groups = []
            batch_scores = []

            # --- 新增：在批次级别获取真实类别 ---
            # data.y 是 one-hot 编码的，我们要转成索引
            current_true_classes = torch.argmax(data.y, dim=1) + 1  # 别忘了 +1
            all_true_classes.extend(current_true_classes.cpu().numpy())
            # --- 结束新增 ---

            for sample_idx in range(self.samples_per_input):
                molecule_list, true_molecule_list, products_list, scores, nll, ell, pred_classes = self.sample_batch(
                    data=data,
                    batch_id=ident,
                    batch_size=to_generate,
                    save_final=to_save,
                    keep_chain=chains_save,
                    number_chain_steps_to_save=self.number_chain_steps_to_save,
                    sample_idx=sample_idx,
                )
                samples.extend(molecule_list)
                batch_groups.append(molecule_list)
                batch_scores.append(scores)

                # --- 新增：只收集第一次采样的预测类别 (Top-1) ---
                if sample_idx == 0:
                    all_pred_classes.extend(pred_classes)
                # --- 结束新增 ---

                if sample_idx == 0:
                    ground_truth.extend(true_molecule_list)

            ident += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

            for mol_idx_in_batch in range(bs):
                mol_samples_group = []
                mol_scores_group = []
                for batch_group, scores_group in zip(batch_groups, batch_scores):
                    mol_samples_group.append(batch_group[mol_idx_in_batch])
                    mol_scores_group.append(scores_group[mol_idx_in_batch])

                assert len(mol_samples_group) == self.samples_per_input
                grouped_samples.append(mol_samples_group)
                grouped_scores.append(mol_scores_group)

        # --- 新增：在所有采样完成后，计算并记录端到端类别准确率 ---
        if len(all_true_classes) == len(all_pred_classes):
            true_tensor = torch.tensor(all_true_classes, device=self.device)
            pred_tensor = torch.tensor(all_pred_classes, device=self.device)

            # 确保 self.val_class_accuracy 在 self.device 上
            self.val_class_accuracy.to(self.device)

            # 注意：类别索引是 1-10，而 torchmetrics 期望 0-9，所以要 -1
            self.val_class_accuracy.update(pred_tensor - 1, true_tensor - 1)

            end_to_end_class_accuracy = self.val_class_accuracy.compute()

            print(
                f"\nEnd-to-End Validation Reaction Type Accuracy (Top-1): {end_to_end_class_accuracy:.4f}")
            self.log('val/end_to_end_class_accuracy',
                     end_to_end_class_accuracy, prog_bar=True)
        else:
            print(
                f"\nWarning: Mismatch in length of true ({len(all_true_classes)}) and predicted ({len(all_pred_classes)}) classes. Cannot compute accuracy.")
        # --- 结束新增 ---

        to_log = compute_retrosynthesis_metrics(
            grouped_samples=grouped_samples,
            ground_truth=ground_truth,
            atom_decoder=self.dataset_info.atom_decoder,
            grouped_scores=grouped_scores,
        )
        for metric_name, metric in to_log.items():
            self.log(metric_name, metric)

        to_log = self.sampling_metrics(samples)
        for metric_name, metric in to_log.items():
            self.log(metric_name, metric)

        self.sampling_metrics.reset()

    def apply_noise(self, X, E, y, X_T, E_T, y_T, node_mask):
        # Sample a timestep t.
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1,
                              size=(X.size(0), 1), device=X.device).float()
        t_float = t_int / self.T
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        # 1. 获取统一的转移矩阵，现在 Qtb.y 是有意义的了
        Qtb = self.transition_model.get_Qt_bar(
            alpha_bar_t=alpha_t_bar, X_T=X_T, E_T=E_T, y_T=y_T, node_mask=node_mask, device=self.device,
        )

        # 2. 对 X, E, y 应用各自的转移矩阵
        # 2a. 图谱 X, E
        probX = (X.unsqueeze(-2) @ Qtb.X).squeeze(-2)
        probE = (E.unsqueeze(-2) @ Qtb.E).squeeze(-2)
        sampled_t_graph = diffusion_utils.sample_discrete_features(
            probX=probX, probE=probE, node_mask=node_mask)
        X_t = F.one_hot(sampled_t_graph.X,
                        num_classes=self.Xdim_output).float()
        E_t = F.one_hot(sampled_t_graph.E,
                        num_classes=self.Edim_output).float()

        # 2b. 类别 y
        #    y 是起点 (随机类别), Qtb.y 是从 y 到 y_t 的转移矩阵
        prob_y = (y.unsqueeze(-2) @ Qtb.y).squeeze(-2)
        sampled_y = torch.multinomial(prob_y, 1).squeeze(-1)
        y_t = F.one_hot(sampled_y, num_classes=self.ydim_output).float()

        # 3. 组合 noisy data
        z_t = utils.PlaceHolder(
            X=X_t, E=E_t, y=y_t).type_as(X_t).mask(node_mask)

        noisy_data = {
            't_int': t_int, 't': t_float, 'beta_t': self.noise_schedule(t_normalized=t_float),
            'alpha_s_bar': self.noise_schedule.get_alpha_bar(t_normalized=(t_int - 1) / self.T),
            'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask
        }
        return noisy_data

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()

        # # --- 打印维度信息 ---
        # print("\n--- Inside MarkovBridge.forward ---")
        # print(
        #     f"Shape of noisy_data['y_t'] (from dataset): {noisy_data['y_t'].shape}")
        # print(
        #     f"Shape of extra_data.y (computed features): {extra_data.y.shape}")
        # # --- 结束打印 ---

        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()

        # # --- 打印维度信息 ---
        # print(f"Shape of final y fed to model: {y.shape}")
        # # 打印模型期望的输入维度
        # expected_dim = self.model.mlp_in_y[0].in_features
        # print(f"Model's mlp_in_y expects input dim: {expected_dim}")
        # print("--- End of MarkovBridge.forward ---\n")
        # # --- 结束打印 ---

        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(
            self,
            data,
            batch_id,
            batch_size,
            keep_chain,
            number_chain_steps_to_save,
            save_final,
            sample_idx,
            save_true_reactants=True,
            use_one_hot=False,
    ):
        """
        :param data
        :param batch_id: int
        :param batch_size: int
        :param number_chain_steps_to_save: int
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param sample_idx: int
        :param save_true_reactants: bool
        :param use_one_hot: convert predictions to one hot before computing transition matrices
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """

        chain_X, chain_E, true_molecule_list, products_list, molecule_list, _, nll, ell, pred_classes = self.sample_chain(
            data=data,
            batch_size=batch_size,
            keep_chain=keep_chain,
            number_chain_steps_to_save=number_chain_steps_to_save,
            save_true_reactants=save_true_reactants,
            use_one_hot=use_one_hot,
        )

        if self.visualization_tools is not None:
            self.visualize(
                chain_X=chain_X,
                chain_E=chain_E,
                true_molecule_list=true_molecule_list,
                products_list=products_list,
                molecule_list=molecule_list,
                sample_idx=sample_idx,
                batch_id=batch_id,
                save_final=save_final
            )

        return molecule_list, true_molecule_list, products_list, [0] * len(molecule_list), nll, ell, pred_classes

    def sample_chain_no_true_no_save(self, data, batch_size, use_one_hot=False):
        # Context product
        product, node_mask = utils.to_dense(
            data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product = product.mask(node_mask)

        # Creating context
        context = product.clone() if self.use_context else None

        # Masks for fixed and modifiable nodes
        fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
        modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
        assert torch.all(fixed_nodes | modifiable_nodes)

        # z_T – starting state (product)
        X, E, y = product.X, product.E, torch.empty(
            (node_mask.shape[0], 0), device=self.device)

        assert (E == torch.transpose(E, 1, 2)).all()

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in tqdm(reversed(range(0, self.T)), total=self.T):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s, node_log_likelihood, edge_log_likelihood = self.sample_p_zs_given_zt(
                s=s_norm,
                t=t_norm,
                X_t=X,
                E_t=E,
                y_t=y,
                X_T=product.X,
                E_T=product.E,
                y_T=product.y,
                node_mask=node_mask,
                context=context,
                use_one_hot=use_one_hot,
            )

            # Masking unchanged part
            if self.fix_product_nodes:
                sampled_s.X = sampled_s.X * modifiable_nodes + product.X * fixed_nodes
                sampled_s = sampled_s.mask(node_mask)

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        molecule_list = utils.create_pred_reactant_molecules(
            X, E, data.batch, batch_size)

        return molecule_list

    def sample_chain(
            self, data, batch_size, keep_chain, number_chain_steps_to_save, save_true_reactants, use_one_hot=False
    ):
        # 1. 准备采样的起点 z_T = (G_P, r_rand)
        #    (在代码的逆向过程中，这对应我们任务的起点)
        bs, device = (data.batch[-1] + 1, data.p_x.device)

        # 1a. 准备图谱部分 G_P
        #     为产物的 y 创建一个0维占位符，因为它本身没有类别信息
        product_y_placeholder = torch.zeros((bs, 0), device=device)
        product, node_mask = utils.to_dense(
            data.p_x, data.p_edge_index, data.p_edge_attr, data.batch, y=product_y_placeholder
        )
        product = product.mask(node_mask)

        # 1b. 准备类别部分 r_rand
        #     这完全采纳了导师的建议：采样的类别一端是随机的
        num_classes = self.ydim_output  # 从模型属性获取类别数，例如 10
        random_indices = torch.randint(0, num_classes, (bs,), device=device)
        y_rand = F.one_hot(random_indices, num_classes=num_classes).float()

        # 2. 初始化采样循环的起始状态为 z_T = (G_P, r_rand)
        X, E, y = product.X, product.E, y_rand

        # 检查维度是否正确
        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps_to_save < self.T

        # 初始化用于保存过程的张量
        chain_X_size = torch.Size(
            (number_chain_steps_to_save, keep_chain, X.size(1)))
        chain_E_size = torch.Size(
            (number_chain_steps_to_save, keep_chain, E.size(1), E.size(2)))
        chain_X = torch.zeros(chain_X_size, device=device)
        chain_E = torch.zeros(chain_E_size, device=device)

        nll = torch.zeros(batch_size, device=device, dtype=torch.float64)
        ell = torch.zeros(batch_size, device=device, dtype=torch.float64)

        # 3. 迭代采样，从 t=T 走到 t=0
        for s_int in tqdm(reversed(range(0, self.T)), total=self.T):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # 调用核心采样步骤，y_t 是上一轮的输出 y (或初始的 y_rand)
            # 这里的 X_T, E_T, y_T 是模型的预测目标，在采样时其实不直接用于计算 z_s
            # 而是用于计算转移矩阵，因此我们传递模型当前的预测目标（即产物和当前y）
            # sample_p_zs_given_zt 内部会用模型预测来更新这个目标
            sampled_s, discrete_sampled_s, node_log_likelihood, edge_log_likelihood = self.sample_p_zs_given_zt(
                s=s_norm,
                t=t_norm,
                X_t=X,
                E_t=E,
                y_t=y,
                X_T=product.X,
                E_T=product.E,
                y_T=y,  # 将当前 y 作为目标传入
                node_mask=node_mask,
                context=product.clone() if self.use_context else None,
                use_one_hot=use_one_hot,
            )

            # Masking aunchanged part (if enabled)
            if self.fix_product_nodes:
                fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
                modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
                assert torch.all(fixed_nodes | modifiable_nodes)

                sampled_s.X = sampled_s.X * modifiable_nodes + product.X * fixed_nodes
                sampled_s = sampled_s.mask(node_mask)

                discrete_sampled_s = sampled_s.clone()
                discrete_sampled_s = discrete_sampled_s.mask(
                    node_mask, collapse=True)

            # 更新状态以进行下一次迭代
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # 保存可视化链
            if keep_chain > 0:
                write_index = (s_int * number_chain_steps_to_save) // self.T
                chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

            nll += node_log_likelihood
            ell += edge_log_likelihood

        # 4. 准备返回值
        pred = sampled_s.clone()  # 保存最终的 one-hot 预测

        # 离散化最终结果
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y_final_prob = sampled_s.X, sampled_s.E, y

        # 从最终的 y 概率分布中得到预测类别
        pred_classes = torch.argmax(y_final_prob, dim=-1) + 1

        # 准备可视化链
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            chain_X = torch.cat(
                [chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat(
                [chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.shape[0] == (number_chain_steps_to_save + 10)

        # 创建分子列表
        true_molecule_list = utils.create_true_reactant_molecules(
            data, batch_size) if save_true_reactants else []
        products_list = utils.create_input_product_molecules(data, batch_size)
        molecule_list = utils.create_pred_reactant_molecules(
            X, E, data.batch, batch_size)

        return (
            chain_X,
            chain_E,
            true_molecule_list,
            products_list,
            molecule_list,
            pred,
            nll.detach().cpu().numpy().tolist(),
            ell.detach().cpu().numpy().tolist(),
            pred_classes.detach().cpu().numpy().tolist(),
        )

    def visualize(
            self,
            chain_X,
            chain_E,
            true_molecule_list,
            products_list,
            molecule_list,
            sample_idx,
            batch_id,
            save_final
    ):
        current_samples_path = os.path.join(
            self.graphs_dir, f'epoch{self.current_epoch}_b{batch_id}')
        current_chains_dir = os.path.join(
            self.chains_dir, f'epoch_{self.current_epoch}')

        if sample_idx == 0:
            # 1. Visualize chains
            num_molecules = chain_X.shape[1]
            for i in range(num_molecules):
                results_path = os.path.join(
                    current_chains_dir, f'molecule_{batch_id + i}')
                os.makedirs(results_path, exist_ok=True)
                self.visualization_tools.visualize_chain(
                    path=results_path,
                    nodes_list=chain_X[:, i, :].numpy(),
                    adjacency_matrix=chain_E[:, i, :].numpy(),
                )

            # 2. Visualize true reactants
            self.visualization_tools.visualize(
                path=current_samples_path,
                molecules=true_molecule_list,
                num_molecules_to_visualize=save_final,
                prefix='true_',
            )

            # 3. Visualize input products
            self.visualization_tools.visualize(
                path=current_samples_path,
                molecules=products_list,
                num_molecules_to_visualize=save_final,
                prefix='input_product_',
            )

        # Visualize predicted reactants
        self.visualization_tools.visualize(
            path=current_samples_path,
            molecules=molecule_list,
            num_molecules_to_visualize=save_final,
            prefix=f'pred_',
            suffix=f'_{sample_idx}'
        )

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, X_T, E_T, y_T, node_mask, context=None, use_one_hot=False):
        """
        Samples from zs ~ p(zs | zt), where zt is the current state and zs is the previous state.
        This function performs one step of the reverse Markov bridge process.
        """
        # Hack: in direct MB we consider flipped time flow for the noise schedule
        bs, n = X_t.shape[:2]
        t_flipped = 1 - t
        beta_t = self.noise_schedule(t_normalized=t_flipped)

        # 1. Use the neural network to predict the final state z_T = (G_R, r_true) from the current state z_t
        noisy_data = {'X_t': X_t, 'E_t': E_t,
                      'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data, context=context)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize the predictions to get probability distributions for the target state
        pred_X = F.softmax(pred.X, dim=-1)
        pred_E = F.softmax(pred.E, dim=-1)
        pred_y = F.softmax(pred.y, dim=-1)

        if use_one_hot:
            x_mask = node_mask.unsqueeze(-1).float()
            e_mask1 = x_mask.unsqueeze(2).float()
            e_mask2 = x_mask.unsqueeze(1).float()
            pred_X = F.one_hot(pred.X.argmax(dim=-1),
                               num_classes=self.Xdim_output).float() * x_mask
            pred_E = F.one_hot(pred.E.argmax(
                dim=-1), num_classes=self.Edim_output).float() * e_mask1 * e_mask2
            pred_y = F.one_hot(pred.y.argmax(dim=-1),
                               num_classes=self.ydim_output).float()

        # 2. Compute the one-step transition matrices Qt(z_s | z_t) using the model's prediction as the target
        Qt = self.transition_model.get_Qt(
            beta_t=beta_t,
            X_T=pred_X,
            E_T=pred_E,
            y_T=pred_y,
            node_mask=node_mask,
            device=self.device,
        )

        # 3. Compute the probability distribution of the previous state z_s, p(z_s | z_t)
        #    This is done by applying the transpose of the transition matrix to the current state z_t
        #    p(z_s) = p(z_t) @ Qt.T

        # 3a. Node transition probabilities
        unnormalized_prob_X = X_t.unsqueeze(-2) @ Qt.X
        unnormalized_prob_X = unnormalized_prob_X.squeeze(-2)
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / \
            torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)

        # 3b. Edge transition probabilities
        E_t_flat = E_t.flatten(start_dim=1, end_dim=2)
        Qt_E_flat = Qt.E.flatten(start_dim=1, end_dim=2)
        unnormalized_prob_E = E_t_flat.unsqueeze(-2) @ Qt_E_flat
        unnormalized_prob_E = unnormalized_prob_E.squeeze(-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / \
            torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        # 3c. Class transition probabilities (now using the same logic)
        unnormalized_prob_y = y_t.unsqueeze(-2) @ Qt.y
        unnormalized_prob_y = unnormalized_prob_y.squeeze(-2)
        unnormalized_prob_y[torch.sum(unnormalized_prob_y, dim=-1) == 0] = 1e-5
        prob_y = unnormalized_prob_y / \
            torch.sum(unnormalized_prob_y, dim=-1, keepdim=True)

        # Sanity checks for probabilities
        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_y.sum(dim=-1) - 1).abs() < 1e-4).all()

        # 4. Sample the discrete state z_s from the computed probability distributions
        sampled_s_graph = diffusion_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask)
        X_s = F.one_hot(sampled_s_graph.X,
                        num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s_graph.E,
                        num_classes=self.Edim_output).float()

        # Sample the class for the next state
        sampled_y_s = torch.multinomial(prob_y, 1).squeeze(-1)
        y_s = F.one_hot(sampled_y_s, num_classes=self.ydim_output).float()

        # Sanity checks for one-hot and symmetry
        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (
            E_t.shape == E_s.shape) and (y_t.shape == y_s.shape)

        # 5. Prepare return values
        # The new continuous state for the next iteration is the sampled one-hot state
        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_s)
        # The discrete state is just the argmax of the one-hot state
        out_discrete = utils.PlaceHolder(
            X=sampled_s_graph.X, E=sampled_s_graph.E, y=sampled_y_s)

        # Likelihood computation for scoring (optional)
        node_log_likelihood = (torch.log(prob_X) +
                               torch.log(pred_X)).sum(dim=(-1, -2))
        edge_log_likelihood = (torch.log(prob_E) + torch.log(pred_E)
                               ).sum(dim=(-1, -2, -3)) / 2  # Divide by 2 for symmetry
        class_log_likelihood = (
            torch.log(prob_y) + torch.log(pred_y)).sum(dim=-1)

        # Combine likelihoods
        nll = node_log_likelihood + class_log_likelihood
        ell = edge_log_likelihood

        return (
            out_one_hot.mask(node_mask).type_as(y_t),
            out_discrete.mask(node_mask, collapse=True).type_as(y_t),
            nll,
            ell,
        )

    def compute_q_zs_given_q_zt(self, z_t, z_T, node_mask, t):
        X_t = z_t.X.to(torch.float32)
        E_t = z_t.E.to(torch.float32)

        # Hack: in direct MB we consider flipped time flow
        bs, n = X_t.shape[:2]
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)

        # Normalize predictions
        X_T = z_T.X.to(torch.float32)  # bs, n, d0
        E_T = z_T.E.to(torch.float32)  # bs, n, n, d0
        y_T = z_T.y

        # Compute transition matrices given prediction
        Qt = self.transition_model.get_Qt(
            beta_t=beta_t,
            X_T=X_T,
            E_T=E_T,
            y_T=y_T,
            node_mask=node_mask,
            device=self.device,
        )  # (bs, n, dx_in, dx_out), (bs, n, n, de_in, de_out)

        # Node transition probabilities
        unnormalized_prob_X = X_t.unsqueeze(-2) @ Qt.X  # bs, n, 1, d_t
        unnormalized_prob_X = unnormalized_prob_X.squeeze(-2)  # bs, n, d_t
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / \
            torch.sum(unnormalized_prob_X, dim=-1,
                      keepdim=True)  # bs, n, d_t-1

        # Edge transition probabilities
        E_T_flat = E_t.flatten(start_dim=1, end_dim=2)  # (bs, N, d_t)
        Qt_E_flat = Qt.E.flatten(start_dim=1, end_dim=2)  # (bs, N, d_t-1, d_t)
        # bs, N, 1, d_t
        unnormalized_prob_E = E_T_flat.unsqueeze(-2) @ Qt_E_flat
        unnormalized_prob_E = unnormalized_prob_E.squeeze(-2)  # bs, N, d_t
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / \
            torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, E_T.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        return prob_X, prob_E

    def compute_p_zs_given_p_zt(self, z_t, pred, node_mask, t):
        p_X_T = F.softmax(pred.X, dim=-1)  # bs, n, d
        p_E_T = F.softmax(pred.E, dim=-1)  # bs, n, n, d

        prob_X = torch.zeros_like(p_X_T)  # bs, n, d
        prob_E = torch.zeros_like(p_E_T)  # bs, n, n, d

        for i in range(self.Xdim_output):
            X_T_i = F.one_hot(torch.ones_like(p_X_T[..., 0]).long(
            ) * i, num_classes=self.Xdim_output).float()
            E_T_i = F.one_hot(torch.zeros_like(
                p_E_T[..., 0]).long(), num_classes=self.Edim_output).float()
            z_T = utils.PlaceHolder(X_T_i, E_T_i)
            prob_X_i, _ = self.compute_q_zs_given_q_zt(
                z_t, z_T, node_mask, t)  # bs, n, d
            prob_X += prob_X_i * p_X_T[..., i].unsqueeze(-1)  # bs, n, d

        for i in range(self.Edim_output):
            X_T_i = F.one_hot(torch.zeros_like(
                p_X_T[..., 0]).long(), num_classes=self.Xdim_output).float()
            E_T_i = F.one_hot(torch.ones_like(p_E_T[..., 0]).long(
            ) * i, num_classes=self.Edim_output).float()
            z_T = utils.PlaceHolder(X_T_i, E_T_i)
            _, prob_E_i = self.compute_q_zs_given_q_zt(
                z_t, z_T, node_mask, t)  # bs, n, n, d
            prob_E += prob_E_i * p_E_T[..., i].unsqueeze(-1)  # bs, n, n, d

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        return prob_X, prob_E

    def compute_extra_data(self, noisy_data, context=None, condition_on_t=True):
        if self.global_step == 1:
            self.print("\n--- Inside compute_extra_data ---")

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), -1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), -1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), -1)

        if context is not None:
            extra_X = torch.cat((extra_X, context.X), -1)
            extra_E = torch.cat((extra_E, context.E), -1)

        if condition_on_t:
            extra_y = torch.cat((extra_y, noisy_data['t']), 1)

        if self.global_step == 1:
            self.print(f"Shape of extra_y: {extra_y.shape}")
            self.print("--- End of compute_extra_data ---\n")

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
