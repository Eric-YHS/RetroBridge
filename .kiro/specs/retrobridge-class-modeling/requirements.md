# Requirements Document

## Introduction

扩展现有的retrobridge模型，使其能够建模联合分布p_theta(G_R, r | G_P)，其中r是反应类别。当前模型只能建模反应物分布p_theta(G_R | G_P)，新功能将增加对反应类别的预测能力，通过引入类别桥接过程来实现分子图生成和类别预测的联合学习。

## Requirements

### Requirement 1

**User Story:** 作为研究人员，我希望模型能够同时预测反应物分子图和反应类别，以便更准确地理解化学反应的完整信息。

#### Acceptance Criteria

1. WHEN 模型接收产物分子图G_P作为输入 THEN 系统SHALL输出反应物分子图G_R和反应类别r的联合分布
2. WHEN 训练时 THEN 系统SHALL使用USPTO50k数据集中的class列（1-10类别）作为监督信号
3. WHEN 模型训练完成 THEN 系统SHALL能够生成符合指定类别的反应物分子图

### Requirement 2

**User Story:** 作为开发者，我希望实现类别桥接过程，以便将类别信息集成到现有的扩散模型框架中。

#### Acceptance Criteria

1. WHEN t=0时 THEN 系统SHALL使用均匀分布(1/K, 1/K, ..., 1/K)作为类别的起点状态，其中K=10
2. WHEN t=T时 THEN 系统SHALL使用one-hot分布作为类别的终点状态，表示真实类别
3. WHEN 前向过程执行时 THEN 系统SHALL通过线性插值计算中间时刻t的类别状态c_t
4. WHEN 计算c_t时 THEN 系统SHALL使用公式P(c_t) = c_0 @ Q_bar_t(class_true)
5. WHEN 反向过程执行时 THEN 系统SHALL从c_t和时间t预测真实类别class_true

### Requirement 3

**User Story:** 作为开发者，我希望修改模型架构以支持类别预测，同时保持原有分子图生成功能。

#### Acceptance Criteria

1. WHEN 训练时 THEN 系统SHALL将输入y从extra_feature扩展为extra_feature+加噪后的class
2. WHEN 模型输出时 THEN 系统SHALL将y的输出维度从0维扩展为10维（对应10个类别）
3. WHEN 处理X和E时 THEN 系统SHALL保持原有的处理方式不变
4. WHEN 推理时 THEN 系统SHALL将输入y从extra_feature扩展为extra_feature+从uniform随机采样的类别

### Requirement 4

**User Story:** 作为研究人员，我希望能够独立监控不同组件的损失，以便更好地理解模型的学习过程。

#### Acceptance Criteria

1. WHEN 计算损失时 THEN 系统SHALL分别计算原有损失和新的类别交叉熵损失
2. WHEN 配置损失权重时 THEN 系统SHALL支持通过yaml文件中的lambda_train参数设置权重
3. WHEN 设置lambda_train时 THEN 系统SHALL使用格式[E权重, y权重]，其中X权重硬编码为1
4. WHEN 训练过程中 THEN 系统SHALL分别记录和显示两个损失的数值
5. WHEN 反向传播时 THEN 系统SHALL使用加权后的总损失进行梯度更新

### Requirement 5

**User Story:** 作为开发者，我希望类别转移矩阵的实现与现有的X和E转移矩阵保持一致，以确保系统的统一性。

#### Acceptance Criteria

1. WHEN 实现类别转移矩阵时 THEN 系统SHALL参考X和E的转移矩阵实现方式
2. WHEN 计算类别转移时 THEN 系统SHALL使用线性插值方法
3. WHEN 采样过程中 THEN 系统SHALL使用相同的时间调度和噪声调度策略
4. WHEN 推理循环时 THEN 系统SHALL按照X、E、y的相同步骤进行迭代更新