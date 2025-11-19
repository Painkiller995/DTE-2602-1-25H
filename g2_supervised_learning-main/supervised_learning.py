import csv
import random as rnd
from typing import Union, cast

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats  # Used for "mode" - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html
from decision_tree_nodes import DecisionTreeBranchNode, DecisionTreeLeafNode
from matplotlib import lines
from numpy.typing import NDArray

# The code below is "starter code" for graded assignment 2 in DTE-2602
# You should implement every method / function which only contains "pass".
# "Helper functions" can be left unedited.
# Feel free to add additional classes / methods / functions to answer the assignment.
# You can use the modules imported above, and you can also import any module included
# in Python 3.10. See https://docs.python.org/3.10/py-modindex.html .
# Using any other modules / external libraries is NOT ALLOWED.


#########################################
#   Data input / prediction evaluation
#########################################

FILE_NAME = "palmer_penguins.csv"

SPECIES_COL_INDEX = 0
SPECIES_MAPPING = {
    "Adelie": 0,
    "Chinstrap": 1,
    "Gentoo": 2,
}

# feature name : column index in dataset (mapping)
FEATURES_DATASET_MAPPING = {
    "bill_length_mm": 2,
    "bill_depth_mm": 3,
    "flipper_length_mm": 4,
    "body_mass_g": 5,
}

FEATURES_DATASET_INDICES = list(FEATURES_DATASET_MAPPING.values())


def read_csv_file(file_name: str, skip_header: bool = True) -> list[list[str]]:
    """
    Read CSV file and return data as list of lists of strings

    Parameters
    ----------
    file_name: str
        Path to CSV file

    Returns
    -------
    data: list[list[str]]
        Data read from CSV file, as list of lists of strings
    """
    with open(file_name, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader)  # Skip header row
        data = [row for row in reader]
    return data


def read_data() -> tuple[NDArray, NDArray]:
    """
    Read data from CSV file, remove rows with missing data, and normalize

    Returns
    -------
    X: NDArray
        Numpy array, shape (n_samples,4), where n_samples is number of rows
        in the dataset. Contains the four numeric columns in the dataset
        (bill length, bill depth, flipper length, body mass).
        Each column (each feature) is normalized by subtracting the column mean
        and dividing by the column std.dev. ("z-score").
        Rows with missing data ("NA") are discarded.
    y: NDarray
        Numpy array, shape (n_samples,)
        Contains integer values (0, 1 or 2) representing the penguin species

    Notes
    -----
    Z-score normalization: https://en.wikipedia.org/wiki/Standard_score .
    """
    raw_csv_rows = read_csv_file(FILE_NAME)

    # Remove rows with missing data ("NA") in any column
    valid_rows = [row for row in raw_csv_rows if "NA" not in row]

    data = np.array(valid_rows)

    # ----------------------------------------------------------
    # X matrix: extract numeric columns and normalize (z-score)
    data_numeric = data[:, FEATURES_DATASET_INDICES].astype(float)
    data_mean = np.mean(data_numeric, axis=0)
    data_std = np.std(data_numeric, axis=0)
    X = (data_numeric - data_mean) / data_std

    # ----------------------------------------------------------
    # y vector: Based on species column, map species names to integers using SPECIES_MAPPING
    species_unique = np.unique(data[:, SPECIES_COL_INDEX])

    if set(species_unique) != set(SPECIES_MAPPING.keys()):
        raise ValueError("Unexpected species in dataset")

    species_to_int = {species: SPECIES_MAPPING[species] for species in species_unique}
    y = np.array([species_to_int[species] for species in data[:, SPECIES_COL_INDEX]])

    return X, y


def convert_y_to_binary(y: NDArray, y_value_true: int) -> NDArray:
    """
    Convert integer valued y to binary (0 or 1) valued vector

    Parameters
    ----------
    y: NDArray
        Integer valued NumPy vector, shape (n_samples,)
    y_value_true: int
        Value of y which will be converted to 1 in output.
        All other values are converted to 0.

    Returns
    -------
    y_binary: NDArray
        Binary vector, shape (n_samples,)
        1 for values in y that are equal to y_value_true, 0 otherwise
    """
    return (y == y_value_true).astype(int)


def train_test_split(
    X: NDArray,
    y: NDArray,
    train_frac: float,
) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
    """
    Shuffle and split dataset into training and testing datasets

    Parameters
    ----------
    X: NDArray
        Dataset, shape (n_samples,n_features)
    y: NDArray
        Values to be predicted, shape (n_samples)
    train_frac: float
        Fraction of data to be used for training

    Returns
    -------
    (X_train,y_train): tuple[NDArray, NDArray]]
        Training dataset
    (X_test,y_test): tuple[NDArray, NDArray]]
        Test dataset
    """

    n_samples, _ = X.shape

    # Generate shuffled array of indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Shuffle X and y
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Find split index
    split_index = int(train_frac * n_samples)

    X_train = X_shuffled[:split_index]
    y_train = y_shuffled[:split_index]

    X_test = X_shuffled[split_index:]
    y_test = y_shuffled[split_index:]

    return (X_train, y_train), (X_test, y_test)


def accuracy(y_pred: NDArray, y_true: NDArray) -> float:
    """
    Calculate accuracy of model based on predicted and true values

    Parameters
    ----------
    y_pred: NDArray
        Numpy array with predicted values, shape (n_samples,)
    y_true: NDArray
        Numpy array with true values, shape (n_samples,)

    Returns
    -------
    accuracy: float
        Fraction of cases where the predicted values
        are equal to the true values. Number in range [0,1]

    # Notes:
    See https://en.wikipedia.org/wiki/Accuracy_and_precision#In_classification
    """
    correct_count = np.sum(y_pred == y_true)
    return correct_count / len(y_true)


##############################
#   Gini impurity functions
##############################


def gini_impurity(y: NDArray) -> float:
    """
    Calculate Gini impurity of a vector

    Parameters
    ----------
    y: NDArray, integers
        1D NumPy array with class labels

    Returns
    -------
    impurity: float
        Gini impurity, scalar in range [0,1)

    # Notes:
    - Wikipedia ref.: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """

    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)

    p = counts / len(y)

    return 1 - np.sum(p**2)


def gini_impurity_reduction(y: NDArray, left_mask: NDArray) -> float:
    """
    Calculate the reduction in mean impurity from a binary split

    Parameters
    ----------
    y: NDArray
        1D numpy array
    left_mask: NDArray
        1D numpy boolean array, True for "left" elements, False for "right"

    Returns
    -------
    impurity_reduction: float
        Reduction in mean Gini impurity, scalar in range [0,0.5]
        Reduction is measured as _difference_ between Gini impurity for
        the original (not split) dataset, and the _weighted mean impurity_
        for the two split datasets ("left" and "right").

    """
    # Total impurity before split
    impurity_before = gini_impurity(y)

    y_left = y[left_mask]
    y_right = y[~left_mask]

    # Sizes of each group
    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)

    impurity_left = gini_impurity(y_left)
    impurity_right = gini_impurity(y_right)
    impurity_after = (n_left / n) * impurity_left + (n_right / n) * impurity_right

    # Reduction = impurity_before - impurity_after
    return impurity_before - impurity_after


def best_split_feature_value(X: NDArray, y: NDArray) -> tuple[float, int, float]:
    """
    Find feature and value "split" that yields highest impurity reduction

    Parameters
    ----------
    X: NDArray
        NumPy feature matrix, shape (n_samples, n_features)
    y: NDArray
        NumPy class label vector, shape (n_samples,)

    Returns
    -------
    impurity_reduction: float
        Reduction in Gini impurity for best split.
        Zero if no split that reduces impurity exists.
    feature_index: int
        Index of X column with best feature for split.
        None if impurity_reduction = 0.
    feature_value: float
        Value of feature in X yielding best split of y
        Dataset is split using X[:,feature_index] <= feature_value
        None if impurity_reduction = 0.

    Notes
    -----
    The method checks every possible combination of feature and
    existing unique feature values in the dataset.
    """

    _, n_features = X.shape

    impurity_reduction: float = -np.inf  # Start with very low value
    best_feature_index: int = -1  # -1 indicates no feature found yet
    best_feature_value: float = np.nan  # NaN indicates no value found yet

    for feature_index in range(n_features):
        feature_values = np.unique(X[:, feature_index])
        for value in feature_values:
            left_mask = X[:, feature_index] <= value
            reduction = gini_impurity_reduction(y, left_mask)
            # Update best split if a better impurity reduction is found
            if reduction > impurity_reduction:
                impurity_reduction = reduction
                best_feature_index = feature_index
                best_feature_value = value

    return impurity_reduction, best_feature_index, best_feature_value


###################
#   Perceptron
###################


class Perceptron:
    """
    Perceptron model for classifying two classes

    Attributes
    ----------
    weights: NDArray
        Array, shape (n_features,), with perceptron weights
    bias: float
        Perceptron bias value
    converged: bool | None
        Boolean indicating if Perceptron has converged during training.
        Set to None if Perceptron has not yet been trained.
    """

    def __init__(self):
        """Initialize perceptron"""
        self.weights: NDArray = np.array([])
        self.bias: float = 0.0
        self.converged: bool | None = None

    def predict_single(self, x: NDArray) -> int:
        """
        Predict / calculate perceptron output for single observation / row x

        Parameters
        ----------
        X: NDArray
            NumPy feature vector, shape (n_features,)

        Returns
        -------
        output: int
            Perceptron output (0 or 1)
        """
        # z = w1*x1 + w2*x2 + ... + wn*xn + b
        z = np.dot(self.weights, x) + self.bias
        return int(z > 0)

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict / calculate perceptron output for data matrix X

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)

        Returns
        -------
        outputs: NDArray
            Perceptron outputs (0 or 1) for each row in X, shape (n_samples,)
        """
        return np.array([self.predict_single(x) for x in X])

    def train(self, X: NDArray, y: NDArray, learning_rate: float, max_epochs: int):
        """
        Fit perceptron to training data X with binary labels y

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray
            NumPy binary class label vector, shape (n_samples,)
        learning_rate: float
            Learning rate for weight updates
        max_epochs: int
            Maximum number of training epochs
        """

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.converged = None

        for epoch in range(max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.predict_single(xi)
                update = learning_rate * (yi - prediction)
                if update != 0:
                    self.weights += update * xi
                    self.bias += update
                    errors += 1
            if errors == 0:
                self.converged = True
                break  # stop training early if converged
        else:
            self.converged = False

    def decision_boundary_slope_intercept(self) -> tuple[float, float]:
        """Calculate slope and intercept for decision boundary line (2-feature data only)
        Calculate slope and intercept for decision boundary line (2-feature data only)

        Returns
        -------
        slope: float
            Slope of the decision boundary line
        intercept: float
            Intercept of the decision boundary line
        """
        if self.weights[1] == 0:
            raise ValueError("Cannot calculate slope when weight[1] is zero")
        slope = -self.weights[0] / self.weights[1]
        intercept = -self.bias / self.weights[1]
        return slope, intercept


####################
#   Decision tree
####################


class DecisionTree:
    """Decision tree model for classification

    Attributes
    ----------
    _root: DecisionTreeBranchNode | None
        Root node in decision tree
    """

    def __init__(self):
        """Initialize decision tree"""
        self._root = None

    def __str__(self) -> str:
        """Return string representation of decision tree (based on binarytree.Node.__str__())"""
        if self._root is not None:
            return str(self._root)
        else:
            return "<Empty decision tree>"

    def fit(self, X: NDArray, y: NDArray):
        """Train decision tree based on labelled dataset

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray, integers
            NumPy class label vector, shape (n_samples,)

        Notes
        -----
        Creates the decision tree by calling _build_tree() and setting
        the root node to the "top" DecisionTreeBranchNode.

        """
        self._root = self._build_tree(X, y)

    def _build_tree(self, X: NDArray, y: NDArray):
        """Recursively build decision tree

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray
            NumPy class label vector, shape (n_samples,)

        Notes
        -----
        - Determines the best possible binary split of the dataset. If no impurity
        reduction can be achieved, a leaf node is created, and its value is set to
        the most common class in y. If a split can achieve impurity reduction,
        a decision (branch) node is created, with left and right subtrees created by
        recursively calling _build_tree on the left and right subsets.

        """
        # Find best binary split of dataset
        impurity_reduction, feature_index, feature_value = best_split_feature_value(X, y)

        # If impurity can't be reduced further, create and return leaf node
        if impurity_reduction == 0:
            leaf_value = scipy.stats.mode(y, keepdims=False)[0]
            return DecisionTreeLeafNode(leaf_value)

        # If impurity _can_ be reduced, split dataset, build left and right
        # branches, and return branch node.
        else:
            left_mask = X[:, feature_index] <= feature_value
            left = self._build_tree(X[left_mask], y[left_mask])
            right = self._build_tree(X[~left_mask], y[~left_mask])
            return DecisionTreeBranchNode(feature_index, feature_value, left, right)

    def predict(self, X: NDArray):
        """Predict class (y vector) for feature matrix X

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)

        Returns
        -------
        y: NDArray, integers
            NumPy class label vector (predicted), shape (n_samples,)
        """
        if self._root is not None:
            return self._predict(X, self._root)
        else:
            raise ValueError("Decision tree root is None (not set)")

    def _predict(
        self,
        X: NDArray,
        node: Union["DecisionTreeBranchNode", "DecisionTreeLeafNode"],
    ) -> NDArray:
        """Predict class (y vector) for feature matrix X

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        node: "DecisionTreeBranchNode" or "DecisionTreeLeafNode"
            Node used to process the data. If the node is a leaf node,
            the data is classified with the value of the leaf node.
            If the node is a branch node, the data is split into left
            and right subsets, and classified by recursively calling
            _predict() on the left and right subsets.

        Returns
        -------
        y: NDArray
            NumPy class label vector (predicted), shape (n_samples,)

        Notes
        -----
        The prediction follows the following logic:

            if the node is a leaf node
                return y vector with all values equal to leaf node value
            else (the node is a branch node)
                split the dataset into left and right parts using node question
                predict classes for left and right datasets (using left and right branches)
                "stitch" predictions for left and right datasets into single y vector
                return y vector (length matching number of rows in X)
        """
        pass


############
#   MAIN
############

if __name__ == "__main__":
    perceptron = Perceptron()
    X = np.array(
        [
            [0.1, 0],
            [0.3, 0.5],
            [0, 0.2],
            [0.4, 0.4],
            [0, 0.3],
            [0.3, 0],
            [0.2, 0.4],
            [0.1, 0.1],
            [0.3, 0.1],
            [0.3, 0.4],
            [0, 0.4],
            [0.1, 0.3],
            [0.2, 0.3],
            [0.5, 0.1],
            [0.3, 0.2],
            [0.5, 1],
            [0.6, 0.9],
            [0.8, 0.9],
            [0.6, 0.5],
            [0.8, 1],
            [0.9, 0.8],
            [1, 0.7],
            [0.5, 0.9],
            [0.5, 1],
            [0.75, 0.6],
            [0.75, 0.8],
            [0.5, 0.7],
        ]
    )

    y = np.array([0] * 15 + [1] * 12)  # First 15 samples are class 0, last 12 samples are class 1
    perceptron.train(X, y, learning_rate=0.3, max_epochs=10)

    y_pred_perceptron = perceptron.predict(X)
    acc_perceptron = accuracy(y_pred_perceptron, y)
    print(f"Perceptron accuracy (class 0 vs rest): {acc_perceptron * 100:.2f}%")
    print("-" * 75)

    print(f"Weights: {perceptron.weights}, Bias: {perceptron.bias}, Converged: {perceptron.converged}")

    plt.scatter(X[:, 0], X[:, 1], c=y)
    slope, intercept = perceptron.decision_boundary_slope_intercept()
    ax = plt.gca()
    x_line = ax.get_xbound()
    y_line = np.array(x_line) * slope + intercept
    plt.plot(x_line, y_line, color="red")
    plt.title("Perceptron Training Data (Class 0 vs Rest)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

if __name__ == "__maintest__":
    # --------------------------------------------
    # Load and prepare dataset
    X, y = read_data()
    print("-" * 75)
    print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

    # --------------------------------------------
    # Split into training and test sets
    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_frac=0.7)
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print("-" * 75)

    # --------------------------------------------
    # Perceptron
    perceptron = Perceptron()
    y_train_binary = convert_y_to_binary(y_train, y_value_true=0)
    perceptron.train(X_train, y_train_binary, learning_rate=0.3, max_epochs=100)

    y_pred_perceptron = perceptron.predict(X_test)
    acc_perceptron = accuracy(y_pred_perceptron, convert_y_to_binary(y_test, y_value_true=0))
    print(f"Perceptron accuracy (class 0 vs rest): {acc_perceptron * 100:.2f}%")
    print("-" * 75)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_binary)
    slope, intercept = perceptron.decision_boundary_slope_intercept()

    ax = plt.gca()
    x_line = ax.get_xbound()
    y_line = np.array(x_line) * slope + intercept
    plt.plot(x_line, y_line, color="red")
    plt.title("Perceptron Training Data (Class 0 vs Rest)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Decision Tree
    # --------------------------------------------
    # decision_tree = DecisionTree()
    # decision_tree.fit(X_train, y_train)
    # print(decision_tree)
