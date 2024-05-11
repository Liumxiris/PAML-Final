import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import mode
from sklearn.metrics import accuracy_score, f1_score

if "processed_df" in st.session_state:
    mbti_data = st.session_state.processed_df
    row_count = st.slider("Select number of rows to view", 1, len(mbti_data), 100)
    st.dataframe(mbti_data.head(row_count))


# Three distance metrics for KNN
def euclidean(v1, v2):
    # Euclidean distance (l2 norm)
    return np.sqrt(np.sum((v1 - v2) ** 2))


def manhattan(v1, v2):
    # Manhattan distance (l1 norm)
    return np.sum(np.abs(v1 - v2))


def minkowski(v1, v2, p=2):
    # Minkowski distance (lp norm)
    return np.sum(np.abs(v1 - v2) ** p) ** (1 / p)


def cosine(v1, v2):
    # Cosine similarity
    return np.dot(v1, v2) / (normalize(v1) * normalize(v2))


def normalize(X):
    means = np.mean(X, axis=0)  # columnwise mean and std
    stds = np.std(X, axis=0) + 1e-7
    X_normalized = (X - means) / (stds)
    return X_normalized


# Get nearest neighbours
def get_neighbours(X, test_row, Y, k, p, metric):
    """
    Returns k nearest neighbors
    """
    distances_ = []
    neigh_class = []

    # Calculate distance to all points in X_train
    for train_row, train_class in zip(X, Y):
        dist = 0
        if metric == "euclidean":
            dist = euclidean(train_row, test_row)
        elif metric == "manhattan":
            dist = manhattan(train_row, test_row)
        elif metric == "minkowski":
            dist = minkowski(train_row, test_row, p)
        elif metric == "cosine":
            dist = minkowski(train_row, test_row, p)
        else:
            raise NameError(
                "Supported metrics are euclidean, manhattan, cosine, and minkowski"
            )
        distances_.append(dist)
        neigh_class.append(train_class)

    # Sort distances
    sorted_ind = np.argsort(distances_, axis=0)  # by rows

    # Identify k nearest neighbours
    sorted_dist = [distances_[i] for i in sorted_ind][:k]
    neighbours_class = [neigh_class[i] for i in sorted_ind][:k]

    return neighbours_class, sorted_dist


def predict(X_train, X_test, Y, k, p, metric):
    """
    Make predictions
    """
    kclusters = []
    distances = []
    # Loop over rows in test set
    for test_row in X_test:
        nearest_neighbours, n_dist = get_neighbours(X_train, test_row, Y, k, p, metric)
        kclusters.append(nearest_neighbours)
        distances.append(n_dist)
    return np.array(kclusters), np.array(distances)


# KNN Hyper-paramters
num_neighhbors = 15  # Number of neighbors
metric = "manhattan"  # Options: 'manhattan', 'minkowski', 'euclidean', 'cosine'
p = 2  # 1 for l1-norm; 2 for l2-norm

X_train = pd.DataFrame(X_train.toarray())
X_test = pd.DataFrame(X_test.toarray())
y_train = pd.DataFrame(y_train.toarray())
y_test = pd.DataFrame(y_test.toarray())


# Use subsets to speed up predicting
n_start_articles = 0
n_end_articles = 200

X_train_subset = X_train[n_start_articles:n_end_articles]
y_train_subset = np.array(y_train[n_start_articles:n_end_articles])

X_test_subset = X_test[n_start_articles:n_end_articles]
y_test_subset = np.array(y_test[n_start_articles:n_end_articles])


neighbours_class, cluster_distances = predict(
    X_train_subset.to_numpy(),
    X_train_subset.to_numpy(),
    y_train_subset,
    num_neighhbors,
    p,
    metric,
)
neighbours_class = neighbours_class[
    :, 1:
]  # removing the closest neighbor which is the point itself
print(
    "The nearest neighbor clusters distances are: \n{}".format(cluster_distances[:5, :])
)
print("The nearest neighbor clusters predictions: \n{}".format(neighbours_class[:5, :]))


# Finding the dominant class in each row of neighbor classes
cluster_predictions, counts = mode(neighbours_class, axis=1)
print(cluster_predictions)
print(y_test_subset[0], cluster_predictions[0])

# showing accuracy and f1_scores
models_accuracy = accuracy_score(y_test_subset, cluster_predictions)
f1_scores = f1_score(y_test_subset, cluster_predictions, average="macro")
print(models_accuracy, f1_scores)
