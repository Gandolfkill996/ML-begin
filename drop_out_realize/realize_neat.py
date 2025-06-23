import torch
from torch import nn
from d2l import torch as d2l



def init_wreights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = 0.01)


if __name__ == "__main__":
    dropout1, dropout2 = 0.2, 0.5
    num_inputs,num_outputs,num_hiddens1,num_hiddens2=784,10,256,256
    net = nn.Sequential(nn.Flatten(), nn.Linear(784,256), nn.ReLU(), nn.Dropout(dropout1), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout2), nn.Linear(256,10))
    net.apply(init_wreights)
    if torch.cuda.device_count() > 1:
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        print(f"Using devices: {devices}")
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    else:
        devices = [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
        net = net.to(devices[0])
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(),lr = lr)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs)



