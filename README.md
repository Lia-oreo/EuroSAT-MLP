# EuroSAT-MLP：基于NumPy手动实现多层感知机的遥感图像分类
本项目基于纯NumPy手动实现MLP模型、激活函数、损失函数与优化器，完成EuroSAT遥感土地覆盖分类任务

---

## 项目文件说明
| 文件名 | 功能 |
|--------|------|
| `model.py` | 实现Linear层、激活函数(ReLU/Sigmoid/Tanh)、MLP模型结构 |
| `train.py` | 模型训练、测试、保存最优权重、绘制训练曲线 |
| `search.py` | 超参数网格搜索（学习率/隐藏层维度/正则化强度） |
| `optim.py` | 实现交叉熵损失函数、SGD优化器 |
| `data.py` | EuroSAT数据集加载、预处理、数据划分 |
| `visualize_and_errors.py` | 错误样本分析、权重可视化、混淆矩阵、类别指标 |
| `compare_activations.py` | 多激活函数训练结果对比 |

---

## 环境依赖
Python 3.8 ~ 3.11
安装依赖命令：
```bash
pip install numpy matplotlib seaborn scikit-learn
```

---

## 运行指令
### 1. 超参数网格搜索（可选）
```bash
python search.py
```

### 2. 训练最优模型
**最优超参数配置**：ReLU激活 + 学习率0.01 + 隐藏层256 + 无正则化 + 训练40轮
```bash
python train.py --activation relu --lr 0.01 --hidden_dim 256 --weight_decay 0.0 --epochs 40
```

### 3. 错误分析与可视化（权重/混淆矩阵/错分样本）
```bash
python visualize_and_errors.py
```

### 4. 多激活函数对比实验
```bash
python compare_activations.py
```

---

## 实验结果
- 模型结构：双层隐藏层MLP
- 测试集准确率：**59.48%**
- 最优激活函数：ReLU
- 最优超参数：`lr=0.01, hidden_dim=256, weight_decay=0.0`

---

##  最优模型权重下载
训练好的模型权重文件（`best_model_relu.npz`）：
https://pan.baidu.com/s/1_ljDbTja46678C-zwXb4fg?pwd=bk42 提取码: bk42


---

