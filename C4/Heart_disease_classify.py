import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

matplotlib.use('TkAgg')

class Heart_disease_classify:
    def __init__(self):
        self.df_heart = pd.read_csv("heart.csv")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def adv(self):

        # print(self.df_heart.head())
        # print(self.df_heart.value_counts())


        X = self.df_heart.drop(['target'],axis=1)
        y = self.df_heart.target.values
        y = y.reshape(-1,1)
        print(X.shape)
        print(y.shape)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        loss_history, weight_history, bias_history = self.gradient_descent(X_train, y_train, weight, bias, alpha, iteration)

    def sigmoid(self,z):
        y_hat = 1/(1+np.exp(z))
        return y_hat

    def loss_function(self,X,y,w,b):
        y_hat = self.sigmoid(np.dot(X,w)+b)
        loss = -((y*np.log(y_hat) + (1-y)*np.log(1-y_hat)))
        cost = np.sum(loss)/X.shape[0]
        return cost

    def plot(self):
        plt.scatter(x=self.df_heart.age[self.df_heart.target==1],
                    y=self.df_heart.thalach[(self.df_heart.target==1)],c="red")

        plt.scatter(x=self.df_heart.age[self.df_heart.target==0],
                    y=self.df_heart.thalach[(self.df_heart.target==0)],marker='^')

        plt.legend(["Disease","No Disease"])
        plt.xlabel('Age')
        plt.ylabel('Heart Rate')
        plt.show()

    def scaler(self, train, test):
        min = train.min(axis=0)
        max = train.max(axis=0)
        gap = max - min
        train -= min
        train /= gap
        test -= min
        test /= gap
        return train, test

    def loss_function(self, X, y, weight, bias):
        y_hat = weight*X + bias
        loss = y_hat - y
        cost = np.sum(loss**2)/(2*len(X))
        return cost

    def gradient_descent(self,X,y,w,b,lr,iter):
        l_history = np.zeros(iter)
        w_history = np.zeros(iter,w.shape[0],w.shape[1])
        b_history = np.zeros(iter)
        for i in range(iter):
            y_hat = self.sigmoid(np.dot(X,w) + b)
            derivate_w = np.dot(X.T, ((y_hat-y)))/X.shape[0]
            derivate_b = np.sum(y_hat-y)/X.shape[0]
            w = w - lr * derivate_w
            b = b - lr * derivate_b
            l_history[i] = self.loss_function(X,y,w,b)
            print("Turn: ", i+1, " current ture loss: ", l_history[i])
            w_history[i] = w
            b_history[i] = b
        return l_history, w_history, b_history

    def logistic_regression(self,X,y,w,b,lr,iter):
        l_history,w_history,b_history = self.gradient_descent(X,y,w,b,lr,iter)
        print("Final loss: ", l_history[-1])
        y_pred = self.predict(X,w_history[-1],b_history[-1])
        traning_acc = 100 - np.mean(np.abs(y_pred-y_train)
        return

    def predict(self,X,w,b):
        z = np.dot(X,w) + b
        y_hat = self.sigmoid(z)
        y_pred = np.zeros((y_hat.shape[0],1))
        for i in range(y_hat.shape[0]):
            if y_hat[i,0] < 0.5:
                y_pred[i,0] = 0
            else:
                y_pred[i,0] = 1
        return y_pred


if __name__ == "__main__":
    new_inst = Heart_disease_classify()
    new_inst.adv()