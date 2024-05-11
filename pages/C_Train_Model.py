import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import mode
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    multilabel_confusion_matrix,
    confusion_matrix,
)
from sklearn.metrics import accuracy_score

if "processed_df" in st.session_state:
    mbti_data = st.session_state.processed_df
    row_count = st.slider("Select number of rows to view", 1, len(mbti_data), 100)
    st.dataframe(mbti_data.head(row_count))

# ----------------------------------------------
# Feature Engineering
# ----------------------------------------------

new_df = st.session_state.processed_df
new_df["posts"] = new_df["posts"].apply(
    lambda x: x[:100] + "..." if len(x) > 100 else x
)
new_df.head()

# Converting MBTI personality into numerical form using Label Encoding
enc = LabelEncoder()
new_df["type of encoding"] = enc.fit_transform(new_df["type"])

target = new_df["type of encoding"]

le_name_mapping = dict(zip(enc.classes_, enc.fit_transform(enc.classes_)))
new_dict = dict([(value, key) for key, value in le_name_mapping.items()])
new_dict


nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
print(stopwords.words("english"))


class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, text):
        return [self.lemmatizer.lemmatize(word) for word in word_tokenize(text)]


# Vectorizing the posts for the model and filtering Stop-words
vect = CountVectorizer(max_features=10000, stop_words="english", tokenizer=Lemmatizer())

# Converting posts (or training or X feature) into numerical form by count vectorization
train = vect.fit_transform(new_df["posts"])

print(train.shape)


# ----------------------------------------------
# Data-splitting: 70-30
# ----------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    train, target, test_size=0.3, stratify=target, random_state=42
)
print((X_train.shape), (y_train.shape), (X_test.shape), (y_test.shape))


# ----------------------------------------------
# KNN
# ----------------------------------------------


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


# Hyper-parameters
num_neighhbors = 15
metric = "manhattan"  # Options: 'manhattan', 'minkowski', 'euclidean', 'cosine'
p = 2  # 1 for l1-norm; 2 for l2-norm

X_train = pd.DataFrame(X_train.toarray())
X_test = pd.DataFrame(X_test.toarray())
# y_train = pd.DataFrame(y_train.toarray())
# y_test = pd.DataFrame(y_test.toarray())


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


# logistic regression with SGD
class LogisticRegression(object):
    def __init__(self, learning_rate=0.001, num_iterations=500):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.likelihood_history = []


    def predict_probability(self, X):
        '''
        Produces probabilistic estimate for P(y_i = +1 | x_i, w)
            Estimate ranges between 0 and 1.
        Input:
            - X: Input features
            - W: weights/coefficients of logistic regression model
            - b: bias or y-intercept of logistic regression classifier
        Output:
            - y_pred: probability of positive product review
        '''
        # Take dot product of feature_matrix and coefficients
        score = np.dot(X, self.W) + self.b

        # Compute P(y_i = +1 | x_i, w) using the link function
        # y_pred = 1. / (1.+np.exp(-score)) + self.b  # this is a bug
        y_pred = 1. / (1. + np.exp(-score))

        return y_pred


    def compute_avg_log_likelihood(self, X, Y, W):
        '''
        Compute the average log-likelihood of logistic regression coefficients

        Input
            - X: subset of features in dataset
            - Y: true sentiment of inputs
            - W: logistic regression weights
        Output
            - lp: log likelihood estimation
        '''
        indicator = (Y == +1)
        scores = np.dot(X, W)
        logexp = np.log(1. + np.exp(-scores))

        # Simple check to prevent overflow
        mask = np.isinf(logexp)
        logexp[mask] = -scores[mask]

        lp = np.sum((indicator - 1) * scores - logexp) / len(X)
        return lp


    def update_weights(self):
        '''
        Compute the logistic regression derivative using
        gradient ascent and update weights self.W

        Inputs: None
        Output: None
        '''
        try:
            # Make a prediction
            y_pred = self.predict(self.X)

            # Bug
            # dW = self.X.T.dot(self.Y-y_pred) / self.num_features
            # db = np.sum(self.Y-y_pred) / self.num_features

            dW = self.X.T.dot(self.Y - y_pred) / self.num_examples
            db = np.sum(self.Y - y_pred) / self.num_examples

            # update weights and bias
            self.b = self.b + self.learning_rate * db
            self.W = self.W + self.learning_rate * dW

            log_likelihood = 0
            # Compute log-likelihood
            log_likelihood += self.compute_avg_log_likelihood(self.X, self.Y, self.W)
            self.likelihood_history.append(log_likelihood)
        except ValueError as err:
            st.write({str(err)})

    def predict(self, X):
        '''
        Hypothetical function  h(x)
        Input:
            - X: Input features
            - W: weights/coefficients of logistic regression model
            - b: bias or y-intercept of logistic regression classifier
        Output:
            - Y: list of predicted classes
        '''
        y_pred = 0
        try:
            scores = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
            y_pred = [-1 if z <= 0.5 else +1 for z in scores]
        except ValueError as err:
            st.write({str(err)})
        return y_pred


    def fit(self, X, Y):
        '''
        Run gradient ascent to fit features to data using logistic regression
        Input:
            - X: Input features
            - Y: list of actual product sentiment classes
            - num_iterations: # of iterations to update weights using gradient ascent
            - learning_rate: learning rate
        Output:
            - W: predicted weights
            - b: predicted bias
            - likelihood_history: history of log likelihood
        '''
        # no_of_training_examples, no_of_features
        self.num_examples, self.num_features = X.shape

        # weight initialization
        self.W = np.zeros(self.num_features)
        self.X = X
        self.Y = Y
        self.b = 0
        self.likelihood_history = []

        # gradient ascent learning
        try:
            for _ in range(self.num_iterations):
                self.update_weights()
        except ValueError as err:
            st.write({str(err)})


    def get_weights(self, model_name):
        '''
        This function prints the coefficients of the trained models

        Input:
            - model_name (list of strings): list of model names including: 'Logistic Regression', 'Stochastic Gradient Ascent with Logistic Regression'
        Output:
            - out_dict: a dicionary contains the coefficients of the selected models, with the following keys:
            - 'Logistic Regression'
            - 'Stochastic Gradient Ascent with Logistic Regression'
        '''
        out_dict = {'Logistic Regression': [],
                    'Stochastic Gradient Ascent with Logistic Regression': []}
        try:
            # set weight for given model_name
            W = np.array([f for f in self.W])
            out_dict[model_name] = self.W
            # Print Coefficients
            st.write('-------------------------')
            st.write('Model Coefficients for ' + model_name)
            num_positive_weights = np.sum(W >= 0)
            num_negative_weights = np.sum(W < 0)
            st.write('* Number of positive weights: {}'.format(num_positive_weights))
            st.write('* Number of negative weights: {}'.format(num_negative_weights))
        except ValueError as err:
            st.write({str(err)})
        return out_dict


class StochasticLogisticRegression(LogisticRegression):
    def __init__(self, num_iterations, learning_rate, batch_size):
        self.likelihood_history = []
        self.batch_size = batch_size

        # invoking the __init__ of the parent class
        LogisticRegression.__init__(self, learning_rate, num_iterations)

    def fit(self, X, Y):
        self.likelihood_history = []

        self.num_examples, self.num_features = X.shape

        self.W = np.zeros(self.num_features)

        permutation = np.random.permutation(len(X))

        self.X = X[permutation, :]
        self.Y = Y[permutation]
        self.b = 0

        i = 0
        for itr in range(self.num_iterations):
            predictions = self.predict_probability(self.X[i:i + self.batch_size, :])

            indicator = (self.Y[i:i + self.batch_size] == +1)

            errors = indicator - predictions

            for j in range(len(self.W)):
                dW = errors.dot(self.X[i:i + self.batch_size, j].T)

                # Subtract the gradient from the weights
                self.W[j] -= self.learning_rate * dW

            lp = self.compute_avg_log_likelihood(self.X[i:i + self.batch_size, :], self.Y[i:i + self.batch_size],
                                                 self.W)

            self.likelihood_history.append(lp)
            if itr <= 15 or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) \
                    or itr % 10000 == 0 or itr == self.num_iterations - 1:
                data_size = len(X)
                print('Iteration %*d: Average log likelihood (of data points in batch [%0*d:%0*d]) = %.8f' % \
                      (int(np.ceil(np.log10(self.num_iterations))), itr, \
                       int(np.ceil(np.log10(data_size))), i, \
                       int(np.ceil(np.log10(data_size))), i + self.batch_size, lp))

            i += self.batch_size
            if i + self.batch_size > len(self.X):
                permutation = np.random.permutation(len(self.X))
                self.X = self.X[permutation, :]
                self.Y = self.Y[permutation]
                i = 0
            self.learning_rate = self.learning_rate / 1.02


sgd_logreg = StochasticLogisticRegression(learning_rate=0.001, num_iterations=100, batch_size=32)
sgd_logreg.fit(X_train.toarray(), np.ravel(y_train))
y_pred_sgd = sgd_logreg.predict(X_test.toarray())
accuracy = accuracy_score(y_test, y_pred_sgd)
print(f"Accuracy: {accuracy * 100.0}%")