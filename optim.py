# optim.py
import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.cache_probs = None
        self.cache_y = None

    def forward(self, logits, y):
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted_logits)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        
        self.cache_probs = probs
        self.cache_y = y
        
        N = logits.shape[0]
        correct_logprobs = -np.log(probs[np.arange(N), y] + 1e-9)
        return np.sum(correct_logprobs) / N

    def backward(self):
        N = self.cache_probs.shape[0]
        grad_logits = self.cache_probs.copy()
        grad_logits[np.arange(N), self.cache_y] -= 1.0
        grad_logits /= N
        return grad_logits

class SGD:
    def __init__(self, model, lr=0.01, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for layer in self.model.layers:
            if hasattr(layer, 'W'):
                grad_W_with_l2 = layer.grad_W + self.weight_decay * layer.W
                layer.W -= self.lr * grad_W_with_l2
                layer.b -= self.lr * layer.grad_b

    def step_lr_decay(self, decay_rate=0.9):
        self.lr *= decay_rate