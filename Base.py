import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


class Base:
    def __init__(self):
        self.wine = pd.read_csv('winequality-red.csv')

        # self.wine = self.wine.drop(labels=self.wine.index[self.wine['quality'] == 5].tolist()[16:])
        # self.wine = self.wine.drop(labels=self.wine.index[self.wine['quality'] == 6].tolist()[15:])
        # self.wine = self.wine.drop(labels=self.wine.index[self.wine['quality'] == 7].tolist()[14:])
        # self.wine = self.wine.drop(labels=self.wine.index[self.wine['quality'] == 4].tolist()[12:])

        self.x = self.wine.drop(['quality'], axis=1)
        self.y = self.wine['quality']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3)

    def show_tree(self):
        print(self.wine.head())

    def show_data(self):
        print(self.y.value_counts())
        print("count: " + str(self.y.shape[0]))
        plt.hist(self.y, bins=int(180 / 5))
        plt.show()
