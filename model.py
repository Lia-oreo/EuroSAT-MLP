import numpy as np

class Layer:
    def forward(self, x): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Linear(Layer):
    def __init__(self, in_features, out_features):
        # He 初始化
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        self.cache_x = None 
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.cache_x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        self.grad_W = np.dot(self.cache_x.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.W.T)

class ReLU(Layer):
    def __init__(self): self.cache_x = None
    def forward(self, x):
        self.cache_x = x
        return np.maximum(0, x)
    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.cache_x <= 0] = 0
        return grad_input

class Sigmoid(Layer):
    def __init__(self): self.output = None
    def forward(self, x):
        x = np.clip(x, -500, 500) 
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output
    def backward(self, grad_output):
        return grad_output * self.output * (1.0 - self.output)

class Tanh(Layer):
    def __init__(self): self.output = None
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    def backward(self, grad_output):
        return grad_output * (1.0 - self.output ** 2)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu'):
        self.layers = []
        act_dict = {'relu': ReLU, 'sigmoid': Sigmoid, 'tanh': Tanh}
        ActLayer = act_dict[activation.lower()]
        
        #3层神经网络结构：输入 -> 隐藏1 -> 隐藏2 -> 输出
        self.layers.append(Linear(input_dim, hidden_dim))
        self.layers.append(ActLayer())
        
        self.layers.append(Linear(hidden_dim, hidden_dim))
        self.layers.append(ActLayer())
        
        self.layers.append(Linear(hidden_dim, output_dim))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)