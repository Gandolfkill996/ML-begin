import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')

class ADV:

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


if __name__ == "__main__":
    new_inst = ADV()
    new_inst.adv()