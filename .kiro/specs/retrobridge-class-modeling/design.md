# Design Document

## Overview

本设计文档描述了如何扩展现有的retrobridge模型，使其能够建模联合分布p_theta(G_R, r | G_P)。核心思想是在现有的分子图扩散过程基础上，增加一个并行的类别扩散过程，通过共享的graph transformer同时学习分子图生成和类别预测。

## Architecture

### 高层架构
```
输入: 产物分子图G_P + 类别噪声c_t
     ↓
Graph Transformer (共享编码器)
     ↓
输出: 反应物分子图G_R + 类别预测r
```

### 类别桥接过程设计

#### 前向过程 (加噪)
- **起点 (t=0)**: 均匀分布 uniform(1/K) where K=10
- **终点 (t=T)**: one-hot编码的真实类别
- **中间状态**: c_t = c_0 @ Q_bar_t(class_true)
- **转移矩阵**: 使用与X、E相同的线性插值方法

#### 反向过程 (去噪)
- 输入: 噪声类别c_t + 时间步t + 产物图G_P
- 输出: 预测的类别分布 (10维softmax)

## Components and Interfaces

### 1. 数据处理组件

#### RetroBridgeDataset 修改
```python
class RetroBridgeDataset(InMemoryDataset):
    def process(self):
        # 现有逻辑保持不变
        # 新增: 读取class列并转换为tensor
        reaction_class = table['class'].values - 1  # 转换为0-9索引
        class_tensor = torch.tensor(reaction_class, dtype=torch.long)
        
        # 在Data对象中添加class字段
        data = Data(
            x=r_x, edge_index=r_edge_index, edge_attr=r_edge_attr, y=y, idx=i,
            p_x=p_x, p_edge_index=p_edge_index, p_edge_attr=p_edge_attr,
            r_smiles=reactants_smi, p_smiles=product_smi,
            reaction_class=class_tensor  # 新增
        )
```

### 2. 模型架构组件

#### MarkovBridge 修改
```python
class MarkovBridge(pl.LightningModule):
    def __init__(self, ...):
        # 现有初始化保持不变
        self.num_classes = 10  # 新增
        # 类别使用简单的线性插值，不需要单独的转移模型
    
    def apply_class_noise(self, true_class, alpha_bar_t):
        """为类别添加噪声 - 使用线性插值"""
        batch_size = true_class.shape[0]
        device = true_class.device
        
        # 创建均匀分布作为起点
        uniform_dist = torch.ones(batch_size, self.num_classes, device=device) / self.num_classes
        
        # 创建one-hot分布作为终点
        one_hot_dist = F.one_hot(true_class, self.num_classes).float()
        
        # 线性插值: (1-alpha_bar_t) * uniform + alpha_bar_t * one_hot
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)  # (bs, 1)
        interpolated_dist = (1 - alpha_bar_t) * uniform_dist + alpha_bar_t * one_hot_dist
        
        # 从插值分布中采样
        sampled_class = torch.multinomial(interpolated_dist, 1).squeeze(-1)
        return F.one_hot(sampled_class, self.num_classes).float()
    
    def compute_class_loss(self, pred_class, true_class):
        """计算类别预测损失"""
        return F.cross_entropy(pred_class, true_class)
```

#### GraphTransformer 修改
```python
class GraphTransformer(nn.Module):
    def __init__(self, ...):
        # 现有初始化保持不变
        # 输出维度修改: y从0维变为10维
        self.mlp_out_y = nn.Sequential(
            nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), 
            act_fn_out,
            nn.Linear(hidden_mlp_dims['y'], 10)  # 修改为10维输出
        )
```

### 3. 损失函数组件

#### 训练损失修改
```python
class TrainLossDiscrete:
    def __init__(self, lambda_train):
        self.lambda_X = 1.0  # 硬编码
        self.lambda_E = lambda_train[0]
        self.lambda_y = lambda_train[1]  # 新增类别损失权重
    
    def __call__(self, ..., pred_y=None, true_y=None):
        # 现有X、E损失计算保持不变
        
        # 新增类别损失计算
        if pred_y is not None and true_y is not None:
            class_loss = F.cross_entropy(pred_y, true_y)
            total_loss += self.lambda_y * class_loss
            
        return total_loss
```

## Data Models

### 输入数据结构
```python
# 训练时
input_data = {
    'X': product_nodes,           # 产物节点特征
    'E': product_edges,           # 产物边特征  
    'y': extra_features + noisy_class,  # 扩展特征 + 噪声类别
    't': timestep,                # 时间步
    'node_mask': mask,           # 节点掩码
    'true_class': reaction_class  # 真实类别 (用于损失计算)
}

# 推理时  
input_data = {
    'X': product_nodes,           # 产物节点特征
    'E': product_edges,           # 产物边特征
    'y': extra_features + uniform_class,  # 扩展特征 + 均匀采样类别
    't': timestep,                # 时间步
    'node_mask': mask            # 节点掩码
}
```

### 输出数据结构
```python
output_data = {
    'X': predicted_reactant_nodes,  # 预测的反应物节点
    'E': predicted_reactant_edges,  # 预测的反应物边
    'y': predicted_class_logits     # 预测的类别logits (10维)
}
```

### 类别状态表示
```python
# t=0: 均匀分布
c_0 = torch.ones(10) / 10  # [0.1, 0.1, ..., 0.1]

# t=T: one-hot真实类别
c_T = F.one_hot(true_class, num_classes=10)  # [0, 0, 1, 0, ...]

# t时刻: 插值状态
c_t = sample_from_transition_matrix(c_0, c_T, t)
```

## Error Handling

### 数据验证
- 验证class列的值范围在1-10之间
- 处理缺失的class值（使用默认值或跳过）
- 验证类别转换的正确性（1-10 → 0-9）

### 模型训练错误处理
- 检查类别损失的数值稳定性
- 监控梯度爆炸/消失问题
- 处理类别不平衡问题

### 推理错误处理
- 验证输出类别概率的有效性（和为1）
- 处理极端概率值（接近0或1）

## Testing Strategy

### 单元测试
1. **类别转移矩阵测试**
   - 验证转移矩阵的概率性质（行和为1）
   - 测试边界条件（t=0, t=T）
   - 验证插值的正确性

2. **损失函数测试**
   - 测试类别损失的计算正确性
   - 验证损失权重的应用
   - 测试梯度计算

3. **数据处理测试**
   - 验证class列的正确读取和转换
   - 测试数据增强的一致性
   - 验证批处理的正确性

### 集成测试
1. **端到端训练测试**
   - 小数据集上的快速训练验证
   - 损失收敛性测试
   - 内存使用测试

2. **推理测试**
   - 类别预测的一致性测试
   - 采样质量评估
   - 性能基准测试

### 性能测试
1. **训练性能**
   - 与原模型的训练速度对比
   - 内存使用对比
   - GPU利用率测试

2. **推理性能**
   - 采样速度测试
   - 批量推理性能
   - 模型大小对比

## Implementation Details

### 配置文件修改
```yaml
# retrobridge.yaml
lambda_train: [5, 1]  # [E权重, y权重]，X权重硬编码为1
num_classes: 10       # 新增类别数量配置
```

### 关键算法实现

#### 类别转移矩阵计算
```python
def get_class_transition_matrix(self, alpha_bar_t, true_class):
    """计算类别转移矩阵"""
    # 使用线性插值: (1-alpha_bar_t) * uniform + alpha_bar_t * one_hot
    uniform_dist = torch.ones(self.num_classes) / self.num_classes
    one_hot_dist = F.one_hot(true_class, self.num_classes).float()
    
    transition_matrix = (1 - alpha_bar_t) * uniform_dist + alpha_bar_t * one_hot_dist
    return transition_matrix
```

#### 类别采样过程
```python
def sample_class_step(self, c_t, t, context):
    """单步类别采样"""
    # 通过模型预测下一步类别分布
    pred_logits = self.model(context, c_t, t).y
    pred_probs = F.softmax(pred_logits, dim=-1)
    
    # 计算转移概率并采样
    transition_probs = self.compute_class_transition(pred_probs, t)
    next_class = torch.multinomial(transition_probs, 1)
    
    return next_class
```