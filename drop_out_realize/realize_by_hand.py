import torch
from torch import nn
from d2l import torch as d2l

dropout1, dropout2=0.2,0.5
class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net,self).__init__()
        self.num_inputs =num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs,num_hiddens1)
        self.lin2= nn.Linear(num_hiddens1,num_hiddens2)
        self.lin3= nn.Linear(num_hiddens2,num_outputs)
        self.relu =nn.ReLU()
    def forward(self,X):
        H1 = self.relu(self.lin1(X.reshape((-1,self.num_inputs))))
        if self.training == True:
            H1 = dropout_layer(H1,dropout1)
        H2= self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2,dropout2)
        out = self.lin3(H2)
        return out

def dropout_layer(X,dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask=(torch.Tensor(X.shape).uniform_(0,1)> dropout).float().to(X.device)
    return mask * X / (1.0-dropout)


if __name__ == "__main__":
    X=torch.arange(16, dtype=torch.float32).reshape((2,8))
    # print(X)
    # print(dropout_layer(X,0.))
    # print(dropout_layer(X,0.5))
    # print(dropout_layer(X,1.))
    num_inputs,num_outputs,num_hiddens1,num_hiddens2=784,10,256,256

    net = Net(num_inputs, num_outputs,num_hiddens1, num_hiddens2)
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
