from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from Base import Base
from Table import table


class DifferentStacking(Base):
    def realization_linear_classifier(self):
        linear_model = LogisticRegression()
        linear_model.fit(self.x_train, self.y_train)

        y_test_prediction = linear_model.predict(self.x_test)
        y_train_prediction = linear_model.predict(self.x_train)
        table.add_row(['Стекинг "Линейный классификатор"', accuracy_score(y_test_prediction, self.y_test) * 100,
                       accuracy_score(y_train_prediction, self.y_train) * 100])

    def realization_naive_baeis_classifier(self):
        Baeis_model = GaussianNB()
        Baeis_model.fit(self.x_train, self.y_train)

        y_test_prediction = Baeis_model.predict(self.x_test)
        y_train_prediction = Baeis_model.predict(self.x_train)
        table.add_row(
            ['Стекинг "Наивный баейсовский классификатор"', accuracy_score(y_test_prediction, self.y_test) * 100,
             accuracy_score(y_train_prediction, self.y_train) * 100])

    def realization_k_nearest_neighbors(self, neighbors=5):
        neighbors_model = KNeighborsClassifier(n_neighbors=neighbors)
        neighbors_model.fit(self.x_train, self.y_train)

        y_test_prediction = neighbors_model.predict(self.x_test)
        y_train_prediction = neighbors_model.predict(self.x_train)
        table.add_row(['Стекинг "k ближайших соседей"', accuracy_score(y_test_prediction, self.y_test) * 100,
                       accuracy_score(y_train_prediction, self.y_train) * 100])

    def realization_stack(self, neighbors=5):
        estimators = [('log', LogisticRegression()), ('bayes', GaussianNB()),
                      ('knn', KNeighborsClassifier(n_neighbors=neighbors))]
        stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
        stack_model.fit(self.x_train, self.y_train)

        y_test_prediction = stack_model.predict(self.x_test)
        y_train_prediction = stack_model.predict(self.x_train)
        table.add_row(['Стекинг "Ансамблёр"', accuracy_score(y_test_prediction, self.y_test) * 100,
                       accuracy_score(y_train_prediction, self.y_train) * 100])
