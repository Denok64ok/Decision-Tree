from Bagging import RandomForest
from Base import Base
from Boosting import CatBoost
from DecisionTree import DecisionTrees
from Stacking import DifferentStacking
from Table import table

Base().show_data()
Base().show_tree()

trees = DecisionTrees()
trees.realization_decision_trees()

RandomForest().realization_random_forest()

stack = DifferentStacking()
stack.realization_linear_classifier()
stack.realization_naive_baeis_classifier()
stack.realization_k_nearest_neighbors()
stack.realization_stack()

CatBoost().realization_Cat_Boost()

print(table)
trees.show()
