from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import tree

from Base import Base
from Table import table


class DecisionTrees(Base):
    def realization_decision_trees(self, max_depth=5):
        decision_model = tree.DecisionTreeClassifier(max_depth=max_depth, criterion='gini')
        decision_model.fit(self.x_train, self.y_train)

        y_test_prediction = decision_model.predict(self.x_test)
        y_train_prediction = decision_model.predict(self.x_train)
        table.add_row(['Алгоритм "Дерево решений"', accuracy_score(y_test_prediction, self.y_test) * 100,
                       accuracy_score(y_train_prediction, self.y_train) * 100])

    def show(self, max_depth=5):
        decision_model = tree.DecisionTreeClassifier(max_depth=max_depth, criterion='gini')
        decision_model.fit(self.x_train, self.y_train)

        plt.figure(figsize=(50, 30))
        tree.plot_tree(decision_tree=decision_model, filled=True, rounded=True, fontsize=5)
        plt.savefig('decision_tree.png', dpi=300)
        plt.show()
