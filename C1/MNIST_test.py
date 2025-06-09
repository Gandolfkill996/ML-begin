import pandas as pd
import numpy as np
from tensorflow.keras.datasets import boston_housing


class MNIST:

    def data_read(self):
        (X_train, y_train), (X_test, y_test) = boston_housing.load_data()
        print(X_train.shape)
        print(X_train[0].shape)
        print(y_train.shape)




if __name__ == "__main__":
    ins = MNIST()
    ins.data_read()