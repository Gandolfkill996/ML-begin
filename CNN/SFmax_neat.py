import torch
from torch import nn
from d2l import torch as d2l

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def get_trainer(lr=0.1):
    return torch.optim.SGD(net.parameters(), lr)

if __name__ == "__main__":
    batch_size=256
    train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    trainer = get_trainer(0.1)
    num_epochs = 10
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=[torch.device('cpu')])
