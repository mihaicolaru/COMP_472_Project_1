# written by:
# Mihai Olaru   40111734
# Amir Cherif   40047635
# Amish Patel   40044279

# TODO: includes
import numpy as np

import pandas as pd

from sklearn import tree
from sklearn import preprocessing

import graphviz

# encoder for non-numerical values in the feature sets
le = preprocessing.LabelEncoder()

# training dataset
training_data = np.array([
    ['yes', 'no', 'no', 'yes', 'some', '$$$', 'no', 'yes', 'french', '0-10', 'yes'],
    ['yes', 'no', 'no', 'yes', 'full', '$', 'no', 'no', 'thai', '30-60', 'no'],
    ['no', 'yes', 'no', 'no', 'some', '$', 'no', 'no', 'burger', '0-10', 'yes'],
    ['yes', 'no', 'yes', 'yes', 'full', '$', 'yes', 'no', 'thai', '10-30', 'yes'],
    ['yes', 'no', 'yes', 'no', 'full', '$$$', 'no', 'yes', 'french', '>60', 'no'],
    ['no', 'yes', 'no', 'yes', 'some', '$$', 'yes', 'yes', 'italian', '0-10', 'yes'],
    ['no', 'yes', 'no', 'no', 'none', '$', 'yes', 'no', 'burger', '0-10', 'no'],
    ['no', 'no', 'no', 'yes', 'yes', '$$', 'yes', 'yes', 'thai', '0-10', 'yes'],
    ['no', 'yes', 'yes', 'no', 'full', '$', 'yes', 'no', 'burger', '>60', 'no'],
    ['yes', 'yes', 'yes', 'yes', 'full', '$$$', 'no', 'yes', 'italian', '10-30', 'no'],
    ['no', 'no', 'no', 'no', 'none', '$', 'no', 'no', 'thai', '0-10', 'no'],
    ['yes', 'yes', 'yes', 'yes', 'full', '$', 'no', 'no', 'burger', '30-60', 'yes'],
])


# build decision tree and print
def build_tree(dataset, filename):
    df2 = pd.DataFrame(dataset,
                   columns=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait'])
    blankIndex=[''] * len(df2)
    df2.index=blankIndex

    X = dataset[:, 0:10]
    y = dataset[:, 10]

    # le = preprocessing.LabelEncoder()
    X[:, 0] = le.fit_transform(X[:, 0])
    X[:, 1] = le.fit_transform(X[:, 1])
    X[:, 2] = le.fit_transform(X[:, 2])
    X[:, 3] = le.fit_transform(X[:, 3])
    X[:, 4] = le.fit_transform(X[:, 4])
    X[:, 5] = le.fit_transform(X[:, 5])
    X[:, 6] = le.fit_transform(X[:, 6])
    X[:, 7] = le.fit_transform(X[:, 7])
    X[:, 8] = le.fit_transform(X[:, 8])
    X[:, 9] = le.fit_transform(X[:, 9])
    y = le.fit_transform(y)

    dtc = tree.DecisionTreeClassifier(criterion="entropy")

    # train model on given attribute matrix
    dtc.fit(X, y)

    print_tree = input("generate tree file? (y/n)\n")

    if print_tree == "y":
        # print trained model
        tree.plot_tree(dtc)

        dot_data = tree.export_graphviz(dtc, out_file=None,
        feature_names=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est'],
        class_names=le.classes_,
        filled=True, rounded=True,
        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(filename)

    return dtc

# classify any number of input feature sets
def predict(input, dtc):
    # input set of attributes
    y_pred = dtc.predict(input)

    # output prediction based on input set
    print("Predicted output: ", le.inverse_transform(y_pred))

# get input feature set from user
def user_input():
    input_set = []
    index = 0
    while 1:
        make_new = input("input new feature set? (y/n)\n")
        if make_new == "y":
            alt = int(input("Is there a suitable alternative nearby? (0/1)\n"))
            bar = int(input("Does the restaurant have a bar in the waiting area? (0/1)\n"))
            fri = int(input("Is it Friday or Saturday? (0/1)\n"))
            hun = int(input("Are you hungry? (0/1)\n"))
            pat = int(input("How many patrons are there inside? (0/1/2)\n"))
            price = int(input("What is the restaurant's price range? (0/1/2)\n"))
            rain = int(input("Is it raining outside? (0/1)\n"))
            res = int(input("Do you have a reservation? (0/1)\n"))
            type = int(input("What kind of restaurant is it? (0/1/2/3)\n"))
            est = int(input("What is the wait time estimate? (0/1/2/3)\n"))

            new_input = [alt, bar, fri, hun, pat, price, rain, res, type, est]
            input_set.insert(index, new_input)
            index = index + 1
            print(input_set)

        else:
            break

    return input_set



# main starts here
tree_name = input("train data? (yes: name/no: n)\n")
if tree_name != "n":
    DT = build_tree(training_data, tree_name)

while 1:
    run = input("run? (y/n)\n")
    if run != "y":
        break

    predict(user_input(), DT)



# Alt (Alternate): other suitable alternatives nearby   0: No   1: Yes
# Bar: has bar waiting area                             0: No   1: Yes
# Fri (Fri/Sat): is Friday or Saturday                  0: No   1: Yes
# Hun (Hungry): is hungry                               0: No   1: Yes
# Pat (Patrons): how many patrons ()                    0: None 1: Some 2: Full
# Price: price range                                    0: $    1: $$   2: $$$
# Rain (Raining): is raining outside                    0: No   1: Yes
# Res (Reservation): reservation was made               0: No   1: Yes
# Type: what kind of restaurant                         0: French   1: Italian  2: Thai 3: Burger
# Est (WaitEstimate): host wait estimate                0: 0-10     1: 10-30    2: 30-60    3: >60
# WillWait (output):                                    0: No   1: Yes

# dataset = np.array([
#     [1, 0, 0, 1, 1, 2, 0, 1, 0, 0, 1],
#     [1, 0, 0, 1, 2, 0, 0, 0, 2, 2, 0],
#     [0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 1],
#     [1, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1],
#     [1, 0, 1, 0, 2, 2, 0, 1, 0, 3, 0],
#     [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
#     [0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0],
#     [0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 1],
#     [0, 1, 1, 0, 2, 0, 1, 0, 3, 3, 0],
#     [1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
#     [1, 1, 1, 1, 2, 0, 0, 0, 3, 2, 1],
# ])
# print(dataset)




# df2 = pd.DataFrame(dataset,
#                    columns=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait'])
# blankIndex=[''] * len(df2)
# df2.index=blankIndex

# X = dataset[:, 0:10]
# y = dataset[:, 10]

# le = preprocessing.LabelEncoder()
# X[:, 0] = le.fit_transform(X[:, 0])
# X[:, 1] = le.fit_transform(X[:, 1])
# X[:, 2] = le.fit_transform(X[:, 2])
# X[:, 3] = le.fit_transform(X[:, 3])
# X[:, 4] = le.fit_transform(X[:, 4])
# X[:, 5] = le.fit_transform(X[:, 5])
# X[:, 6] = le.fit_transform(X[:, 6])
# X[:, 7] = le.fit_transform(X[:, 7])
# X[:, 8] = le.fit_transform(X[:, 8])
# X[:, 9] = le.fit_transform(X[:, 9])
# y = le.fit_transform(y)

# dtc = tree.DecisionTreeClassifier(criterion="entropy")

# # train model on given attribute matrix
# dtc.fit(X, y)


# # print trained model
# tree.plot_tree(dtc)

# dot_data = tree.export_graphviz(dtc, out_file=None,
# feature_names=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est'],
# class_names=le.classes_,
# filled=True, rounded=True,
# special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render("mytree1")


# # input set of attributes
# y_pred = dtc.predict([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]])

# # output prediction based on input set
# print("Predicted output: ", le.inverse_transform(y_pred))








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
