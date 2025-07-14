import torch
import torch.nn as nn
import torch.nn.functional as F

class ReactionClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim1: int = 256, hidden_dim2: int = 128, dropout_rate: float = 0.3):
        """
        一个简单的多层感知机 (MLP) 分类器。

        参数:
        input_dim (int): 输入特征的维度。
                         (Graph Transformer y 的维度 + 摩根指纹的位数)
        num_classes (int): 要分类的类别数量 (即反应类型的数量)。
        hidden_dim1 (int): 第一个隐藏层的维度。
        hidden_dim2 (int): 第二个隐藏层的维度。
        dropout_rate (float): Dropout的比率。
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1) # 批量归一化层 (可选)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2) # 批量归一化层 (可选)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        # 权重初始化 (可选, 但有时有帮助)
        # self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        分类器的前向传播。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)。

        返回:
        torch.Tensor: 输出的 logits，形状为 (batch_size, num_classes)。
        """
        x = self.fc1(x)
        # x = self.bn1(x) # 如果使用 BatchNorm，请取消注释
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        # x = self.bn2(x) # 如果使用 BatchNorm，请取消注释
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x) # 输出 logits，不需要再接激活函数，因为 F.cross_entropy 会处理
        return x

    def _initialize_weights(self):
        # 一个简单的权重初始化示例
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)