import torch
from torch import nn
from d2l import torch as d2l

def two_layer(num_inputs, num_outputs, num_hiddens):
    # try set from randn to zeros
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True))
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens,num_outputs, requires_grad=True))
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    return [W1, b1, W2, b2]

def relu(X):
    a= torch.zeros_like(X)
    return torch.max(X,a)

def net(X, num_inputs, params):
    X=X.reshape((-1,num_inputs))
    H=relu(X @ params[0] + params[1])
    return (H @ params[2] + params[3])

def set_params(batch_size, num_epochs, lr):
    return batch_size, num_epochs, lr


if __name__ == "__main__":
    batch_size, num_epochs, lr = set_params(256,10,0.1)
    train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
    num_inputs, num_outputs, num_hiddens = 784,10,256
    params = two_layer(num_inputs, num_outputs, num_hiddens)
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(params,lr = lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)