import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

from data import EuroSATDataLoader
from model import MLP
from optim import CrossEntropyLoss, SGD

def get_batches(X, Y, batch_size):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]

def save_model(model, filepath):
    """保存模型中有权重的层的 W 和 b"""
    params = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'W'):
            params[f'W_{i}'] = layer.W
            params[f'b_{i}'] = layer.b
    np.savez(filepath, **params)

def load_hyperparam_results():
    """加载网格搜索的最优参数"""
    try:
        res = np.load('hyperparam_search_results.npz', allow_pickle=True)
        return res['best_params'].item()
    except:
        return {'lr': 0.01, 'hidden_dim': 128, 'wd': 1e-4}

def main(args):
    np.random.seed(args.seed)
    
    if not os.path.exists(args.data_dir):
        raise ValueError(f"数据路径不存在: {args.data_dir}")
    loader = EuroSATDataLoader(args.data_dir)
    X, Y, classes = loader.load_data()
    
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=args.seed)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=args.seed)
    
    #加载最优超参数
    best_params = load_hyperparam_results()
    lr = args.lr if args.lr is not None else best_params['lr']
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else best_params['hidden_dim']
    weight_decay = args.weight_decay if args.weight_decay is not None else best_params['wd']
    
    #模型初始化
    model = MLP(input_dim=12288, hidden_dim=hidden_dim, output_dim=10, activation=args.activation)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model, lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        #训练阶段
        model_loss, correct, total = 0.0, 0, 0
        for batch_X, batch_Y in get_batches(X_train, Y_train, args.batch_size):
            logits = model.forward(batch_X)
            loss = criterion.forward(logits, batch_Y)
            model.backward(criterion.backward())
            optimizer.step()
            
            model_loss += loss * batch_X.shape[0]
            preds = np.argmax(logits, axis=1)
            correct += np.sum(preds == batch_Y)
            total += batch_X.shape[0]
        
        # 学习率衰减
        if args.lr_decay:
            optimizer.step_lr_decay(decay_rate=args.decay_rate)
        
        train_loss = model_loss / total
        train_acc = correct / total
        
        #验证阶段
        val_logits = model.forward(X_val)
        val_loss = criterion.forward(val_logits, Y_val)
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = np.sum(val_preds == Y_val) / len(Y_val)
        
        # 记录历史
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        
        # 打印日志
        if (epoch+1) % args.print_freq == 0:
            print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, f'best_model_{args.activation}.npz')

    #测试阶段
    model = MLP(input_dim=12288, hidden_dim=hidden_dim, output_dim=10, activation=args.activation)
    params = np.load(f'best_model_{args.activation}.npz')
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'W'):
            layer.W = params[f'W_{i}']
            layer.b = params[f'b_{i}']
    
    test_logits = model.forward(X_test)
    test_preds = np.argmax(test_logits, axis=1)
    test_acc = np.sum(test_preds == Y_test) / len(Y_test)
    
    #绘制实验报告图表
    plt.figure(figsize=(15, 5))
    
    # Loss 曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Loss Curve ({args.activation})'); plt.xlabel('Epoch'); plt.legend()
    
    # Accuracy 曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'Accuracy Curve ({args.activation})'); plt.xlabel('Epoch'); plt.legend()
    
    # 混淆矩阵
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(Y_test, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'report_figures_{args.activation}.png')
    plt.show()
    
    #保存训练历史
    np.savez(f'train_history_{args.activation}.npz', **history)

if __name__ == '__main__':
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='MLP训练EuroSAT数据集')
    parser.add_argument('--data_dir', type=str, default='./EuroSAT_RGB', help='数据路径')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率（优先于网格搜索结果）')
    parser.add_argument('--hidden_dim', type=int, default=None, help='隐藏层维度（优先于网格搜索结果）')
    parser.add_argument('--weight_decay', type=float, default=None, help='权重衰减（优先于网格搜索结果）')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'], help='激活函数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--lr_decay', action='store_true', default=True, help='是否启用学习率衰减')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='学习率衰减率')
    parser.add_argument('--print_freq', type=int, default=1, help='日志打印频率')
    
    args = parser.parse_args()
    main(args)