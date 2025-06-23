import torch
import os
import pandas as pd
import random
# from d2l import torch as d2l
import utils

def data_example():
    os.makedirs(os.path.join('..','data'), exist_ok=True)
    data_file = os.path.join('..','data', 'house_tiny.csv')
    with open(data_file, 'w') as f:
        f.write('NumRooms, Alley,Price\n')
        f.write('3, Pave,127500\n')
        f.write('2, NA,106000\n')
        f.write('4, NA,178100\n')
        f.write('NA, NA,140000\n')
    return data_file

def read_data():
    data = pd.read_csv(os.path.join('..','data', 'house_tiny.csv'))
    print(data)
    return data

def generate_puts():
    data = read_data()
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:,2]
    inputs = inputs.fillna(inputs.mean())
    # inputs = pd.get_dummies(inputs,dummy_na=True)
    print(inputs)
    return inputs, outputs

def test():
    x=torch.arange(4.0)
    print(x)
    x.requires_grad_(True)
    print(x.grad)
    y = 2 * torch.dot(x,x)
    print(y)
    y.backward()
    print(x.grad)


def synthetic_data(w, b, num_examples):
    # generate y = Xw + b + bios
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,y.shape)
    return X, y.reshape((-1,1))


def test1():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 10)
    print("features: ", features)
    print("labels: ", labels)
    return features, labels

def linreg(X, w, b):
    # linear regression model
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat-y.reshape(y_hat.shape))**2 / 2

def sgd(params, lr, batch_size):
    # smaall batch random gradient descent
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def lintrain(lr, num_epochs, batch_size):
    net = linreg
    loss = squared_loss
    features, labels = test1()
    for epoch in range(num_epochs):
        for X, y in utils.data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y) # X, and y small batch loss

            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch{epoch+1}, loss {float(train_l.mean()):f}')

if __name__ == "__main__":
    # batch_size = 10
    # features, labels = test1()
    # for X, y in utils.data_iter(batch_size, features, labels):
    #     print(X, '\n',y)
    #     break

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    lintrain(0.03, 10, 10)