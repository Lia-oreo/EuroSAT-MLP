# compare_activations.py (多激活函数对比)
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 加载不同激活函数的训练历史
    activations = ['relu', 'sigmoid', 'tanh']
    histories = {}
    for act in activations:
        try:
            histories[act] = np.load(f'train_history_{act}.npz', allow_pickle=True)
        except:
            print(f"未找到 {act} 的训练历史，请先运行: python train.py --activation {act}")
            return
    
    # 绘制对比图表
    plt.figure(figsize=(12, 10))
    
    # 1. Loss 对比
    plt.subplot(2, 1, 1)
    for act in activations:
        plt.plot(histories[act]['train_loss'], label=f'{act} Train Loss')
        plt.plot(histories[act]['val_loss'], label=f'{act} Val Loss', linestyle='--')
    plt.title('Loss Comparison (Different Activations)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. Accuracy 对比
    plt.subplot(2, 1, 2)
    for act in activations:
        plt.plot(histories[act]['train_acc'], label=f'{act} Train Acc')
        plt.plot(histories[act]['val_acc'], label=f'{act} Val Acc', linestyle='--')
    plt.title('Accuracy Comparison (Different Activations)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('activation_comparison.png')
    plt.show()
    
    # 打印最终指标对比
    print("=== 不同激活函数最终指标对比 ===")
    for act in activations:
        train_acc = histories[act]['train_acc'][-1]
        val_acc = histories[act]['val_acc'][-1]
        print(f"{act.upper()} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

if __name__ == '__main__':
    main()