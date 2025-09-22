import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import MeanSquaredError, Accuracy
from src.metrics.abstract_metrics import CrossEntropyMetric, ProbabilisticCrossEntropyMetric


from pdb import set_trace


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TrainLoss(nn.Module):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.train_node_mse = NodeMSE()
        self.train_edge_mse = EdgeMSE()
        self.train_y_mse = MeanSquaredError()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y):
        mse_X = self.train_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.train_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.train_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y
        to_log = {
            'train_loss/batch_mse': mse.detach(),
            'train_loss/node_MSE': self.train_node_mse.compute(),
            'train_loss/edge_MSE': self.train_edge_mse.compute(),
            'train_loss/y_mse': self.train_y_mse.compute()
        }

        return mse, to_log

    def reset(self):
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()


class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.y_accuracy = Accuracy(task='multiclass', num_classes=10)  # 类别准确率
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, 10) -- 类别预测logits
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs,) -- 类别索引
        log : boolean. """
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        
        # 处理类别损失
        if true_y.numel() > 0 and pred_y.numel() > 0:
            # 检查输入的有效性
            if torch.isnan(pred_y).any() or torch.isinf(pred_y).any():
                print(f"Warning: pred_y contains NaN or Inf values")
                loss_y = torch.tensor(0.0, device=pred_y.device)
            else:
                # 计算交叉熵损失
                loss_y = F.cross_entropy(pred_y, true_y, reduction='mean')
                
                # 更新y_loss指标（用于记录）
                # 需要将true_y转换为one-hot格式以匹配CrossEntropyMetric的期望输入
                true_y_onehot = F.one_hot(true_y, num_classes=10).float()
                self.y_loss.update(pred_y, true_y_onehot)
                
                # 更新类别准确率
                self.y_accuracy.update(pred_y, true_y)
        else:
            loss_y = torch.tensor(0.0, device=masked_pred_X.device)

        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y

    def compute_metrics(self):
        return {
            'X_CE': self.node_loss.compute(),
            'E_CE': self.edge_loss.compute(),
            'y_CE': self.y_loss.compute(),
            'y_accuracy': self.y_accuracy.compute(),  # 添加类别准确率
        }

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss, self.y_accuracy]:
            metric.reset()


class TrainLossVLB(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = ProbabilisticCrossEntropyMetric()
        self.edge_loss = ProbabilisticCrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E):
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0

        return loss_X + self.lambda_train[0] * loss_E

    def compute_metrics(self):
        return {
            'X_CE': self.node_loss.compute(),
            'E_CE': self.edge_loss.compute(),
        }

    def reset(self):
        for metric in [self.node_loss, self.edge_loss]:
            metric.reset()
