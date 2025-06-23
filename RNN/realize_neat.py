import torch
from torch import nn
from d2l import torch as d2l


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal(m.weight, std=0.01)

def set_params(batch_size, num_epochs, lr):
    return batch_size, num_epochs, lr

if __name__ == "__main__":
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),nn.Linear(256,10))
    net.apply(init_weights)
    batch_size, num_epochs, lr = set_params(256, 10, 0.1)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(),lr = lr)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
