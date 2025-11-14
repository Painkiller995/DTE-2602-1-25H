import numpy as np
import pytest
import sklearn.datasets
import supervised_learning as sl


# ----------
# Fixtures
# ----------
@pytest.fixture
def small_2class_test_data():
    """2D, 2-class dataset where one feature separates the classes"""
    X = np.transpose(np.array([[2, 3, 2, 3, 2, 3, 2, 3], [3, 2, 2, 4, 6, 8, 6, 7]]))
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    # Feature 0 is "mixed" across the two labels
    # Feature 1 has low values for label 1 and high values for label 2
    return (X, y)


@pytest.fixture
def gaussian_cluster_test_data_2class():
    centers = np.array([[0, 0], [1, 1]])
    X, y = sklearn.datasets.make_blobs(  # type: ignore
        centers=centers, cluster_std=0.2, n_samples=100, random_state=42
    )
    return (X, y, centers)


@pytest.fixture
def gaussian_cluster_test_data_4class():
    """2D, 4-class dataset with Gaussian distributed "blobs" """
    centers = np.array([[-1, 0], [1, 1], [0.5, -1], [1.5, -1]])
    X, y = sklearn.datasets.make_blobs(  # type: ignore
        centers=centers,
        cluster_std=[0.4, 0.4, 0.2, 0.2],
        n_samples=[50, 30, 15, 15],
        random_state=42,
    )
    return (X, y, centers)


# ----------
#   Tests
# ----------


def test_gini_impurity(gaussian_cluster_test_data_4class):
    """Test Gini impurity for several different y vectors"""
    assert sl.gini_impurity(np.array([1, 1, 1])) == 0
    assert sl.gini_impurity(np.array([1, 1, 2, 2])) == 0.5
    assert sl.gini_impurity(np.array([0, 1, 1, 1, 1, 1, 1, 1])) == 0.21875
    assert sl.gini_impurity(np.array([i // 10 for i in range(100)])) == pytest.approx(
        0.9
    )
    _, y, _ = gaussian_cluster_test_data_4class
    assert sl.gini_impurity(y) == pytest.approx(0.681818, rel=1e-4)


def test_gini_impurity_reduction():
    """Test Gini impurity reduction for simple 2-class vectors"""
    assert (
        sl.gini_impurity_reduction(
            y=np.array([1, 2, 1, 2]), left_mask=np.array([1, 0, 1, 0], dtype=bool)
        )
        == 0.5
    )
    assert (
        sl.gini_impurity_reduction(
            y=np.array([1, 2, 1, 2]), left_mask=np.array([1, 1, 0, 0], dtype=bool)
        )
        == 0
    )
    assert sl.gini_impurity_reduction(
        y=np.array([1, 1, 2, 1, 2, 2]),
        left_mask=np.array([1, 1, 1, 0, 0, 0], dtype=bool),
    ) == pytest.approx(0.055555, rel=1e-4)
    # Imbalanced classes, test weighted mean:
    assert sl.gini_impurity_reduction(
        y=np.array([2, 2, 1, 1, 1, 2, 1]),
        left_mask=np.array([1, 1, 0, 0, 1, 0, 0], dtype=bool),
    ) == pytest.approx(0.085034, rel=1e-4)


def test_best_split_feature_value(
    small_2class_test_data, gaussian_cluster_test_data_4class
):
    """Test detection of "best question" for 2-class and 4-class datasets"""
    X_small, y_small = small_2class_test_data
    assert sl.best_split_feature_value(X_small, y_small) == (0.5, 1, 4)
    X, y, _ = gaussian_cluster_test_data_4class
    assert sl.best_split_feature_value(X, y) == pytest.approx(
        ((0.34090909, 0, -0.36831487))
    )


def test_decisiontree_predict_1(small_2class_test_data):
    X, y = small_2class_test_data
    dt_classifier = sl.DecisionTree()
    dt_classifier.fit(X, y)
    assert dt_classifier._root.feature_index == 1  # type: ignore
    assert dt_classifier._root.feature_value == 4  # type: ignore
    assert dt_classifier._root.left.y_value == 1  # type: ignore
    assert dt_classifier._root.right.y_value == 2  # type: ignore


def test_decisiontree_predict_2(gaussian_cluster_test_data_4class):
    """Fit decision tree to Gaussian blob data and predict classes for blob center points"""
    dt_classifier = sl.DecisionTree()
    X, y, X_centers = gaussian_cluster_test_data_4class
    dt_classifier.fit(X, y)
    assert np.all(dt_classifier.predict(X_centers) == np.array([0, 1, 2, 3]))
