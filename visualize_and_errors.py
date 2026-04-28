import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from data import EuroSATDataLoader
from model import MLP

def visualize_weights(model_filepath):
    """可视化第一层隐藏层的权重"""
    data = np.load(model_filepath)
    W1 = data['W_0'] 
    
    hidden_dim = W1.shape[1]
    
    # 挑选前 16 个神经元的权重来可视化
    num_to_show = min(16, hidden_dim)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    for i in range(num_to_show):
        # 提取第 i 个神经元的权重，形状 (12288,)
        weight_vector = W1[:, i]
        
        # 重新整形为图片尺寸 (64, 64, 3)
        weight_img = weight_vector.reshape(64, 64, 3)
        
        # 将权重归一化到 0-1 之间以便于用 RGB 显示
        w_min, w_max = np.min(weight_img), np.max(weight_img)
        weight_img_normalized = (weight_img - w_min) / (w_max - w_min + 1e-9)
        
        ax = axes[i // 4, i % 4]
        ax.imshow(weight_img_normalized)
        ax.axis('off')
        ax.set_title(f'Neuron {i}')
        
    plt.suptitle("First Layer Weights (Reshaped to 64x64x3)", fontsize=16)
    plt.tight_layout()
    plt.savefig('weight_visualization.png')
    plt.show()

def load_best_model(model_filepath, input_dim=12288, output_dim=10, hidden_dim=128):
    """加载保存的最优模型权重到MLP"""
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, activation='relu')
    params = np.load(model_filepath)
    # 恢复权重
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'W'):
            layer.W = params[f'W_{i}']
            layer.b = params[f'b_{i}']
    return model

def visualize_error_samples(X_test, Y_test, Y_pred, classes, num_samples=10):
    """可视化错误分类的样本"""
    # 找到错误样本的索引
    error_indices = np.where(Y_pred != Y_test)[0]
    # 随机选num_samples个
    selected_indices = np.random.choice(error_indices, min(num_samples, len(error_indices)), replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx, ax in zip(selected_indices, axes):
        # 恢复图像形状 (64,64,3)
        img = X_test[idx].reshape(64, 64, 3)
        true_label = classes[Y_test[idx]]
        pred_label = classes[Y_pred[idx]]
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=8)
    
    plt.suptitle("Misclassified Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig('error_samples.png')
    plt.show()

def analyze_class_performance(Y_test, Y_pred, classes):
    """分析各类别的Precision/Recall/F1，并可视化"""
    report = classification_report(Y_test, Y_pred, target_names=classes, output_dict=True)
    for cls in classes:
        print(f"\n类别 {cls}:")
        print(f"  Precision: {report[cls]['precision']:.4f}")
        print(f"  Recall:    {report[cls]['recall']:.4f}")
        print(f"  F1-Score:  {report[cls]['f1-score']:.4f}")
    
    f1_scores = [report[cls]['f1-score'] for cls in classes]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=classes, y=f1_scores)
    plt.xticks(rotation=45, ha='right')
    plt.title('F1-Score per Class')
    plt.ylabel('F1-Score')
    plt.tight_layout()
    plt.savefig('class_f1_scores.png')
    plt.show()
    
    #混淆矩阵
    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Best Model)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_best.png')
    plt.show()

def main():
    loader = EuroSATDataLoader("./EuroSAT_RGB")
    X, Y, classes = loader.load_data()
    from sklearn.model_selection import train_test_split
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
    
    try:
        search_results = np.load('hyperparam_search_results.npz', allow_pickle=True)
        best_params = search_results['best_params'].item()
        hidden_dim = best_params['hidden_dim']
    except:
        hidden_dim = 128
    
    model = load_best_model('best_model_relu.npz', hidden_dim=hidden_dim)
    
    #预测测试集
    Y_pred = np.argmax(model.forward(X_test), axis=1)
    
    visualize_weights('best_model_relu.npz')
    
    visualize_error_samples(X_test, Y_test, Y_pred, classes)
    
    analyze_class_performance(Y_test, Y_pred, classes)

if __name__ == '__main__':
    main()