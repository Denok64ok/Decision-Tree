from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from Base import Base
from Table import table


class RandomForest(Base):
    def realization_random_forest(self, max_depth=3):
        forest_model = RandomForestClassifier(n_estimators=50, max_depth=max_depth)
        forest_model.fit(self.x_train, self.y_train)

        y_test_prediction = forest_model.predict(self.x_test)
        y_train_prediction = forest_model.predict(self.x_train)
        table.add_row(['Бэггинг "Случайные леса"', accuracy_score(y_test_prediction, self.y_test) * 100,
                       accuracy_score(y_train_prediction, self.y_train) * 100])
