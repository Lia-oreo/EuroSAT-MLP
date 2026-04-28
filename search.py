import numpy as np
from sklearn.model_selection import train_test_split
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

def train_short(model, criterion, optimizer, X_train, Y_train, X_val, Y_val, epochs=5, batch_size=64):
    best_val_acc = 0.0
    for epoch in range(epochs):
        model_loss, correct, total = 0.0, 0, 0
        for batch_X, batch_Y in get_batches(X_train, Y_train, batch_size):
            logits = model.forward(batch_X)
            loss = criterion.forward(logits, batch_Y)
            model.backward(criterion.backward())
            optimizer.step()
            
            model_loss += loss * batch_X.shape[0]
            preds = np.argmax(logits, axis=1)
            correct += np.sum(preds == batch_Y)
            total += batch_X.shape[0]
        
        optimizer.step_lr_decay(decay_rate=0.95)
        val_logits = model.forward(X_val)
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = np.sum(val_preds == Y_val) / len(Y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    return best_val_acc

def main():
    np.random.seed(42)
    # 加载数据
    loader = EuroSATDataLoader("./EuroSAT_RGB")
    X, Y, classes = loader.load_data()

    #采样 15% 数据训练
    X, _, Y, _ = train_test_split(X, Y, test_size=0.85, random_state=42)
    
    # 划分数据集
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
    
    learning_rates = [0.01, 0.001]
    hidden_dims = [128, 256]
    weight_decays = [0.0, 1e-4]
    batch_size = 64
    epochs = 5  

    best_global_acc = 0.0
    best_params = {}
    search_results = []

    for lr in learning_rates:
        for hd in hidden_dims:
            for wd in weight_decays:
                model = MLP(input_dim=12288, hidden_dim=hd, output_dim=10, activation='relu')
                criterion = CrossEntropyLoss()
                optimizer = SGD(model, lr=lr, weight_decay=wd)
                
                val_acc = train_short(model, criterion, optimizer, X_train, Y_train, X_val, Y_val, epochs, batch_size)
                search_results.append({'lr': lr, 'hidden_dim': hd, 'wd': wd, 'val_acc': val_acc})
                
                if val_acc > best_global_acc:
                    best_global_acc = val_acc
                    best_params = {'lr': lr, 'hidden_dim': hd, 'wd': wd, 'val_acc': val_acc}

    # 输出结果
    print(f"最佳超参数: {best_params}")
    print(f"最佳准确率: {best_global_acc:.4f}")
    
    np.savez('hyperparam_search_results.npz', 
             results=search_results, best_params=best_params, best_acc=best_global_acc)

if __name__ == '__main__':
    main()