# import numpy as np

# class DecisionTreeClassifier:
#     def __init__(self):
#         self.tree = None

#     def calculate_entropy(self, data):
#         class_labels = data[:, -1]
#         unique_labels, label_counts = np.unique(class_labels, return_counts=True)
#         total_instances = len(class_labels)
#         probabilities = label_counts / total_instances
#         entropy = -np.sum(probabilities * np.log2(probabilities))
#         return entropy

#     def split_data(self, data, feature_index):
#         feature_values = np.unique(data[:, feature_index])
#         splits = {}
#         for value in feature_values:
#             splits[value] = data[data[:, feature_index] == value]
#         return splits

#     def choose_best_feature(self, data):
#         num_features = data.shape[1] - 1
#         entropy = self.calculate_entropy(data)
#         best_info_gain = 0
#         best_feature = None
#         for feature_index in range(num_features):
#             splits = self.split_data(data, feature_index)
#             new_entropy = 0
#             for value, split_data in splits.items():
#                 probability = len(split_data) / len(data)
#                 new_entropy += probability * self.calculate_entropy(split_data)
#             info_gain = entropy - new_entropy
#             if info_gain > best_info_gain:
#                 best_info_gain = info_gain
#                 best_feature = feature_index
#         return best_feature

#     def create_leaf_node(self, data):
#         class_labels = data[:, -1]
#         unique_labels, label_counts = np.unique(class_labels, return_counts=True)
#         majority_label = unique_labels[np.argmax(label_counts)]
#         return majority_label

#     def create_tree(self, data):
#         if len(np.unique(data[:, -1])) == 1:
#             return self.create_leaf_node(data)

#         if data.shape[1] == 1:
#             return self.create_leaf_node(data)

#         best_feature = self.choose_best_feature(data)
#         splits = self.split_data(data, best_feature)
#         decision_tree = {best_feature: {}}

#         for value, split_data in splits.items():
#             decision_tree[best_feature][value] = self.create_tree(split_data)

#         return decision_tree

#     def fit(self, X, y):
#         data = np.column_stack((X, y))
#         self.tree = self.create_tree(data)

#     def predict_instance(self, instance, tree):
#         if isinstance(tree, str):
#             return tree
#         feature_index = list(tree.keys())[0]
#         feature_value = instance[feature_index]
#         subtree = tree[feature_index][feature_value]
#         return self.predict_instance(instance, subtree)

#     def predict(self, X):
#         predictions = []
#         for instance in X:
#             prediction = self.predict_instance(instance, self.tree)
#             predictions.append(prediction)
#         return np.array(predictions)


# data = np.array([
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 'A'],
#     [2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'],
#     [3, 4, 5, 6, 7, 8, 9, 10, 11, 'B'],
#     [4, 5, 6, 7, 8, 9, 10, 11, 12, 'B'],
#     [5, 6, 7, 8, 9, 10, 11, 12, 13, 'B'],
# ])


# tree_builder = DecisionTreeClassifier()

# entropy = tree_builder.calculate_entropy(data)
# print(entropy)

# tree = tree_builder.create_tree(data)
# print(tree)


import numpy as np

class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    def calculate_entropy(self, data):
        class_labels = data[:, -1]
        unique_labels, label_counts = np.unique(class_labels, return_counts=True)
        total_instances = len(class_labels)
        probabilities = label_counts / total_instances
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def evaluate_splits(self, data):
        num_features = data.shape[1] - 1
        entropy = self.calculate_entropy(data)
        best_info_gain = 0
        best_feature = None
        for feature_index in range(num_features):
            feature_values = np.unique(data[:, feature_index])
            new_entropy = 0
            for value in feature_values:
                split_data = data[data[:, feature_index] == value]
                probability = len(split_data) / len(data)
                new_entropy += probability * self.calculate_entropy(split_data)
            info_gain = entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature_index
        return best_feature

    def split_data(self, data, feature_index):
        feature_values = np.unique(data[:, feature_index])
        splits = {}
        for value in feature_values:
            splits[value] = data[data[:, feature_index] == value]
        return splits

    def create_leaf_node(self, data):
        class_labels = data[:, -1]
        unique_labels, label_counts = np.unique(class_labels, return_counts=True)
        majority_label = unique_labels[np.argmax(label_counts)]
        return majority_label

    def create_tree(self, data):
        # needs revision
        if len(np.unique(data[:, -1])) == 1:
            return self.create_leaf_node(data)

        if data.shape[1] == 1:
            return self.create_leaf_node(data)

        best_feature = self.evaluate_splits(data)
        splits = self.split_data(data, best_feature)
        decision_tree = {best_feature: {}}

        for value, split_data in splits.items():
            decision_tree[best_feature][value] = self.create_tree(split_data)

        return decision_tree

    def fit(self, X, y):
        data = np.column_stack((X, y))
        self.tree = self.create_tree(data)

    def classify_instance(self, instance, tree):
        # questionable code
        if isinstance(tree, str):
            return tree
        feature_index = list(tree.keys())[0]
        feature_value = instance[feature_index]
        if feature_value not in tree[feature_index]:
            return None
        subtree = tree[feature_index][feature_value]
        return self.classify_instance(instance, subtree)

    def predict(self, X):
        predictions = []
        for instance in X:
            prediction = self.classify_instance(instance, self.tree)
            predictions.append(prediction)
        return np.array(predictions)


X = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [2, 3, 4, 5, 6, 7, 8, 9, 10],
    [3, 4, 5, 6, 7, 8, 9, 10, 11],
    [4, 5, 6, 7, 8, 9, 10, 11, 12],
    [5, 6, 7, 8, 9, 10, 11, 12, 13],
])

y = np.array(['A', 'A', 'B', 'B', 'B'])

clf = DecisionTreeClassifier()
clf.fit(X, y)

new_data = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [2, 3, 4, 5, 6, 7, 8, 9, 10],
    [3, 4, 5, 6, 7, 8, 9, 10, 11],
])

predictions = clf.predict(new_data)
print("Predictions:", predictions)
