import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')

class Classification:

    def adv(self):
        df_ads = pd.read_csv("advertising.csv")
        # print(df_ads.head())

        # create Heatmap based on corr between data and sale
        # sns.heatmap(df_ads.corr(), cmap="YlGnBu", annot = True)

        # create scatter plot
        # sns.pairplot(df_ads,
        #              x_vars=['wechat', 'weibo', 'others'],
        #              y_vars='sales',
        #              height=4, aspect=1, kind='scatter')
        # plt.show()

        X = np.array(df_ads.wechat)
        y = np.array(df_ads.sales)
        X = X.reshape(len(X),1)
        y = y.reshape(len(y),1)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
        X_train, X_test = self.scaler(X_train, X_test)
        y_train, y_test = self.scaler(y_train, y_test)

        plt.plot(X_train, y_train, 'r.', label='Training data')
        plt.xlabel('wechat')
        plt.ylabel('sales')
        plt.legend()
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
            y_hat = sigmoid(np.dot(X,w) + b)
            loss = -(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
            derivate_w = np.dot(X.T, ((y_hat-y)))/X.shape[0]
            derivate_b = np.sum(y_hat-y)/X.shape[0]
            w = w - lr * derivate_w
            b = b - lr * derivate_b
            l_history[i] = loss_function(X,y,w,b)
            print("Turn: ", i+1, " current ture loss: ", l_history[i])
            w_history[i] = w
            b_history[i] = b
        return l_history, w_history, b_history


if __name__ == "__main__":
    new_inst = Classification()
    new_inst.adv()