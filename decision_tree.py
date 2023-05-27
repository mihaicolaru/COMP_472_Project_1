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
dataset = np.array([
    ['Yes', 'No', 'No', 'Yes', 'Some', '$$$', 'No', 'Yes', 'French', '0-10', 'Yes'],
    ['Yes', 'No', 'No', 'Yes', 'Full', '$$', 'No', 'No', 'Thai', '30-60', 'No'],
    ['No', 'Yes', 'No', 'No', 'Some', '$', 'No', 'No', 'Burger', '0-10', 'Yes'],
    ['Yes', 'No', 'Yes', 'Yes', 'Full', '$', 'Yes', 'No', 'Thai', '10-30', 'Yes'],
    ['Yes', 'No', 'Yes', 'No', 'Full', '$$$', 'No', 'Yes', 'French', '>60', 'No'],
    ['No', 'Yes', 'No', 'Yes', 'Some', '$$', 'Yes', 'Yes', 'Italian', '0-10', 'Yes'],
    ['No', 'Yes', 'No', 'No', 'None', '$', 'Yes', 'No', 'Burger', '0-10', 'No'],
    ['No', 'No', 'No', 'Yes', 'Some', '$$', 'Yes', 'Yes', 'Thai', '0-10', 'Yes'],
    ['No', 'Yes', 'Yes', 'No', 'Full', '$', 'Yes', 'No', 'Burger', '>60', 'No'],
    ['Yes', 'Yes', 'Yes', 'Yes', 'Full', '$$$', 'No', 'Yes', 'Italian', '10-30', 'No'],
    ['No', 'No', 'No', 'No', 'None', '$', 'No', 'No', 'Thai', '0-10', 'No'],
    ['Yes', 'Yes', 'Yes', 'Yes', 'Full', '$', 'No', 'No', 'Burger', '30-60', 'Yes'],
])
print(dataset)

df2 = pd.DataFrame(dataset,
                   columns=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait'])
blankIndex=[''] * len(df2)
df2.index=blankIndex

df2

X = dataset[:, 0:10]
y = dataset[:, 10]

le = preprocessing.LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
y = le.fit_transform(y)

dtc = tree.DecisionTreeClassifier(criterion="entropy")

# dtc.fit(X, y)

# y_pred = dtc.predict([])
# print("Predicted output: ", le.inverse_transform(y_pred))
# tree.plot_tree(dtc)

# dot_data = tree.export_graphviz(dtc, out_file=None,
# feature_names=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est'],
# class_names=le.classes_,
# filled=True, rounded=True)
# graph = graphviz.Source(dot_data)
# graph.render("mytree1")


# example 2 in the tutorial
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.6, random_state=0)
# create and print the decision tree
dtc = tree.DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train, y_train)
tree.plot_tree(dtc)

dot_data = tree.export_graphviz(dtc, out_file=None,
feature_names=iris.feature_names,
class_names=iris.target_names,
filled=True, rounded=True,
special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iristree")

y_pred = dtc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# this is all stuff taken from the tutorials
