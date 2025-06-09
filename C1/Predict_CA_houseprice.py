import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class C1:

    def data_read(self):
        print(1)
        # ssl_context = ssl.create_default_context(cafile=certifi.where())
        # df_housing = pd.read_csv("https://raw.githubusercontent.com/huangjia2019/house/master/house.csv", storage_options={'ssl_context': ssl_context})
        df_housing = pd.read_csv("https://raw.githubusercontent.com/huangjia2019/house/master/house.csv")
        # print(df_housing.head)
        return df_housing

    def preprocess(self):
        data = self.data_read()
        X = data.drop("median_house_value",axis=1)
        y = data.median_house_value
        return X, y

    def split_data(self):
        marked_data = self.preprocess()
        X = marked_data[0]
        y = marked_data[1]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test

    def model(self):
        splited_data = self.split_data()
        X_train = splited_data[0]
        y_train = splited_data[2]
        model = LinearRegression()
        model.fit(X_train,y_train)
        return model

    def test(self):
        model = self.model()
        splited_data = self.split_data()
        X_test = splited_data[1]
        y_test = splited_data[3]
        y_pred = model.predict(X_test)
        print("real value:", y_test)
        print("predict value:", y_pred)
        print("model score:", model.score(X_test,y_test))




if __name__ == "__main__":
    ins = C1()
    ins.test()
