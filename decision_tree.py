# written by:
# Mihai Olaru   40111734
# add your names and id

# TODO: includes
import numpy as np

import pandas as pd

from sklearn import tree
from sklearn import preprocessing

import graphviz

# TODO: functions

# TODO: main

# Alt (Alternate): other suitable alternatives nearby   0: No   1: Yes
# Bar: has bar waiting area                             0: No   1: Yes
# Fri (Fri/Sat): is Friday or Saturday                  0: No   1: Yes
# Hun (Hungry): is hungry                               0: No   1: Yes
# Pat (Patrons): how many patrons ()                    0: None 1: Some 2: Full
# Price: price range                                    0: $    1: $$   2: $$$
# Rain (Raining): is raining outsidn                    0: No   1: Yes
# Res (Reservation): reservation was made               0: No   1: Yes
# Type: what kind of restaurant                         0: French   1: Italian  2: Thai 3: Burger
# Est (WaitEstimate): host wait estimate                0: 0-10     1: 10-30    2: 30-60    3: >60
# WillWait (output):                                    0: No   1: Yes

dataset = np.array([
    [1, 0, 0, 1, 1, 2, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 2, 0, 0, 0, 2, 2, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 1],
    [1, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1],
    [1, 0, 1, 0, 2, 2, 0, 1, 0, 3, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 1],
    [0, 1, 1, 0, 2, 0, 1, 0, 3, 3, 0],
    [1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [1, 1, 1, 1, 2, 0, 0, 0, 3, 2, 1],
])
# print(dataset)

df2 = pd.DataFrame(dataset,
                   columns=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait'])
blankIndex=[''] * len(df2)
df2.index=blankIndex

X = dataset[:, 0:10]
y = dataset[:, 10]

le = preprocessing.LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
y = le.fit_transform(y)

dtc = tree.DecisionTreeClassifier(criterion="entropy")

# train model on given attribute matrix
dtc.fit(X, y)


# print trained model
tree.plot_tree(dtc)

dot_data = tree.export_graphviz(dtc, out_file=None,
feature_names=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est'],
# class_names=le.classes_,
filled=True, rounded=True,
special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("mytree1")


# input set of attributes
y_pred = dtc.predict([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]])

# output prediction based on input set
print("Predicted output: ", le.inverse_transform(y_pred))








# example 2 in the tutorial
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# # load the Iris dataset
# iris = load_iris()
# X, y = iris.data, iris.target
# X_train, X_test, y_train, y_test = train_test_split(
# X, y, test_size=0.6, random_state=0)
# # create and print the decision tree
# dtc = tree.DecisionTreeClassifier(criterion="entropy")
# dtc.fit(X_train, y_train)
# tree.plot_tree(dtc)

# dot_data = tree.export_graphviz(dtc, out_file=None,
# feature_names=iris.feature_names,
# class_names=iris.target_names,
# filled=True, rounded=True,
# special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render("iristree")

# y_pred = dtc.predict(X_test)
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))

# from sklearn.metrics import confusion_matrix
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
