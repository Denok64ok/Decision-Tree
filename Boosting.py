from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
from Base import Base
from Table import table


class CatBoost(Base):
    def realization_Cat_Boost(self):
        Boost_model = CatBoostClassifier(iterations=20, depth=3, learning_rate=1, loss_function='MultiClass',
                                         verbose=True)
        Boost_model.fit(self.x_train, self.y_train)

        y_test_prediction = Boost_model.predict(self.x_test)
        y_train_prediction = Boost_model.predict(self.x_train)
        table.add_row(['Бустинг "Градиентный бустинг"', accuracy_score(y_test_prediction, self.y_test) * 100,
                       accuracy_score(y_train_prediction, self.y_train) * 100])

        vis = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(self.y_test, y_test_prediction, labels=Boost_model.classes_),
            display_labels=Boost_model.classes_)
        vis.plot()
        plt.show()
