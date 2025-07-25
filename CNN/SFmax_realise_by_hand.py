import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# print(X.sum(0, keepdim=True))
# print(X.sum(1, keepdim=True))
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# print(y_hat[[0, 1], y])



class Accumulator:
    """在~n~个变量上累加。"""
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data =[a+ float(b) for a,b in zip(self.data, args)]
    def reset(self):
        self.data =[0.0]* len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

class Animator:
    def __init__(self,xlabel=None,ylabel=None,legend=None,xlim=None,ylim=None,xscale='linear',yscale='linear',
    fmts=('-','m--','g-.','r:'),nrows=1,ncols=1,figsize=(3.5,2.5)):
        if legend is None:
            legend =[]
        d2l.use_svg_display()
        self.fig,self.axes = d2l.plt.subplots(nrows,ncols,figsize=figsize)
        if nrows * ncols ==1:
            self.axes =[self.axes,]
        self.config_axes = lambda:d2l.set_axes(self.axes[0],xlabel,ylabel,xlim,ylim,xscale,yscale,legend)
        self.X,self.Y,self.fmts = None,None,fmts

    def add(self,x,y):
        if not hasattr(y,"__len__"):
            y =[y]
        n=len(y)

def softmax(X):
    x_exp = torch.exp(X)
    partition = x_exp.sum(1, keepdim=True)
    return x_exp / partition

def net(X):
    # W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat,y):
    """计算预测正确的数量。"""
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat =y_hat.argmax(axis=1)
    cmp =y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter):
    """计算在指定数据集上模型的精度。"""
    if isinstance(net,torch.nn.Module):
        net.eval() #将模型设置为评估模式
    metric =Accumulator(2) #正确预测数、预测总数
    for X,y in data_iter:
        metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric =Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)
        l= loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l)* len(y), accuracy(y_hat,y),y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.size().numel())
    print("metric: ", metric)
    print("metric[0]: ", metric[0])
    print("metric[1]: ", metric[1])
    print("metric[2]: ", metric[2])

    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net,train_iter,test_iter, loss, num_epochs,updater):
    animator = Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3],legend=['train loss','train acc','test acc'])
    for epoch in range(num_epochs):
        train_metrics= train_epoch_ch3(net,train_iter, loss, updater)
        test_acc=evaluate_accuracy(net,test_iter)
        animator.add(epoch +1,train_metrics +(test_acc,))
        train_loss,train_acc = train_metrics

def updater(batch_size, lr=0.1):
    return d2l.sgd([W, b], lr, batch_size)

def predict_ch3(net,test_iter,n=6):
    """预测标签(定义见第3章)"""
    for X,y in test_iter:
        break
    trues =d2l.get_fashion_mnist_labels(y)
    preds= d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles =[true + '\n'+ pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n,28,28)),1,n,titles=titles[0:n])



if __name__ == "__main__":

    cross_entropy(y_hat, y)
    accuracy(y_hat, y) / len(y)
    evaluate_accuracy(net,test_iter)
    num_epochs = 10
    print(train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater))
    print(predict_ch3(net,test_iter))