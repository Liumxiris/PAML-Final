import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import mode
from scipy.sparse import issparse
from numpy import log, dot, exp, shape
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


class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, text):
        return [self.lemmatizer.lemmatize(word) for word in word_tokenize(text)]


class LogisticRegression_jamie(object):
    def sigmoid(self, z):
        sig = 1 / (1 + exp(-z))
        return sig

    def initialize(self, X):
        # Check if bias term is already included in X
        if issparse(X):
            X = X.toarray()
        if X.shape[1] > 1 and (X[:, 0] == 1).all():
            weights = np.zeros((X.shape[1], 1))
            return weights, X
        else:
            weights = np.zeros((X.shape[1] + 1, 1))
            X = np.c_[np.ones((X.shape[0], 1)), X]
            return weights, X

    def fit(self, X, y, alpha=None, iter=None):  # alpha=learning eate, iteration
        weights, X = self.initialize(X)

        def cost(theta):
            z = dot(X, theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1 - y).T.dot(log(1 - self.sigmoid(z)))
            cost = -((cost1 + cost0)) / len(y)
            return cost

        cost_list = np.zeros(
            iter,
        )

        y_numpy = y.to_numpy()
        y_reshaped = np.reshape(y_numpy, (-1, 1))

        # ///////////

        for i in range(iter):

            # weights = weights - alpha*dot(X.T,self.sigmoid(dot(X,weights))-np.reshape(y,(len(y),1)))
            sigmoid_output = self.sigmoid(np.dot(X, weights))
            error = sigmoid_output - y_reshaped
            gradient = np.dot(X.T, error)
            weights_update = alpha * gradient
            weights -= weights_update

            cost_list[i] = cost(weights)
        self.weights = weights
        return cost_list

    def predict(self, X):
        z = dot(self.initialize(X)[1], self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i > 0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis


# logistic regression with SGD
class LogisticRegression(object):
    def __init__(self, learning_rate=0.001, num_iterations=500):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.likelihood_history = []

    def predict_probability(self, X):
        """
        Produces probabilistic estimate for P(y_i = +1 | x_i, w)
            Estimate ranges between 0 and 1.
        Input:
            - X: Input features
            - W: weights/coefficients of logistic regression model
            - b: bias or y-intercept of logistic regression classifier
        Output:
            - y_pred: probability of positive product review
        """
        # Take dot product of feature_matrix and coefficients
        score = np.dot(X, self.W) + self.b

        # Compute P(y_i = +1 | x_i, w) using the link function
        # y_pred = 1. / (1.+np.exp(-score)) + self.b  # this is a bug
        y_pred = 1.0 / (1.0 + np.exp(-score))

        return y_pred

    def compute_avg_log_likelihood(self, X, Y, W):
        """
        Compute the average log-likelihood of logistic regression coefficients

        Input
            - X: subset of features in dataset
            - Y: true sentiment of inputs
            - W: logistic regression weights
        Output
            - lp: log likelihood estimation
        """
        indicator = Y == +1
        scores = np.dot(X, W)
        logexp = np.log(1.0 + np.exp(-scores))

        # Simple check to prevent overflow
        mask = np.isinf(logexp)
        logexp[mask] = -scores[mask]

        lp = np.sum((indicator - 1) * scores - logexp) / len(X)
        return lp

    def update_weights(self):
        """
        Compute the logistic regression derivative using
        gradient ascent and update weights self.W

        Inputs: None
        Output: None
        """
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
        """
        Hypothetical function  h(x)
        Input:
            - X: Input features
            - W: weights/coefficients of logistic regression model
            - b: bias or y-intercept of logistic regression classifier
        Output:
            - Y: list of predicted classes
        """
        y_pred = 0
        try:
            scores = 1 / (1 + np.exp(-(X.dot(self.W) + self.b)))
            y_pred = [-1 if z <= 0.5 else +1 for z in scores]
        except ValueError as err:
            st.write({str(err)})
        return y_pred

    def fit(self, X, Y):
        """
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
        """
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
        """
        This function prints the coefficients of the trained models

        Input:
            - model_name (list of strings): list of model names including: 'Logistic Regression', 'Stochastic Gradient Ascent with Logistic Regression'
        Output:
            - out_dict: a dicionary contains the coefficients of the selected models, with the following keys:
            - 'Logistic Regression'
            - 'Stochastic Gradient Ascent with Logistic Regression'
        """
        out_dict = {
            "Logistic Regression": [],
            "Stochastic Gradient Ascent with Logistic Regression": [],
        }
        try:
            # set weight for given model_name
            W = np.array([f for f in self.W])
            out_dict[model_name] = self.W
            # Print Coefficients
            st.write("-------------------------")
            st.write("Model Coefficients for " + model_name)
            num_positive_weights = np.sum(W >= 0)
            num_negative_weights = np.sum(W < 0)
            st.write("* Number of positive weights: {}".format(num_positive_weights))
            st.write("* Number of negative weights: {}".format(num_negative_weights))
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

        self.X = X.to_numpy()[permutation, :]
        self.Y = Y[permutation]
        self.b = 0

        i = 0
        for itr in range(self.num_iterations):
            predictions = self.predict_probability(self.X[i : i + self.batch_size, :])

            indicator = self.Y[i : i + self.batch_size] == +1

            errors = indicator - predictions

            for j in range(len(self.W)):
                dW = errors.dot(self.X[i : i + self.batch_size, j].T)

                # Subtract the gradient from the weights
                self.W[j] -= self.learning_rate * dW

            lp = self.compute_avg_log_likelihood(
                self.X[i : i + self.batch_size, :],
                self.Y[i : i + self.batch_size],
                self.W,
            )

            self.likelihood_history.append(lp)
            if (
                itr <= 15
                or (itr <= 1000 and itr % 100 == 0)
                or (itr <= 10000 and itr % 1000 == 0)
                or itr % 10000 == 0
                or itr == self.num_iterations - 1
            ):
                data_size = len(X)
                print(
                    "Iteration %*d: Average log likelihood (of data points in batch [%0*d:%0*d]) = %.8f"
                    % (
                        int(np.ceil(np.log10(self.num_iterations))),
                        itr,
                        int(np.ceil(np.log10(data_size))),
                        i,
                        int(np.ceil(np.log10(data_size))),
                        i + self.batch_size,
                        lp,
                    )
                )

            i += self.batch_size
            if i + self.batch_size > len(self.X):
                permutation = np.random.permutation(len(self.X))
                self.X = self.X[permutation, :]
                self.Y = self.Y[permutation]
                i = 0
            self.learning_rate = self.learning_rate / 1.02


if "processed_df" in st.session_state:
    new_df = st.session_state.processed_df

    # ----------------------------------------------
    # Feature Engineering
    # ----------------------------------------------

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

    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")
    print(stopwords.words("english"))

    # Vectorizing the posts for the model and filtering Stop-words
    vect = CountVectorizer(
        max_features=10000, stop_words="english", tokenizer=Lemmatizer()
    )

    # Converting posts (or training or X feature) into numerical form by count vectorization
    train = vect.fit_transform(new_df["posts"])
    # train = vetorize_posts(new_df["posts"])

    print(train.shape)

    # ----------------------------------------------
    # Data-splitting: 70-30
    # ----------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=0.3, stratify=target, random_state=42
    )
    print((X_train.shape), (y_train.shape), (X_test.shape), (y_test.shape))

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
                dist = cosine(train_row, test_row)
            else:
                raise NameError(
                    "Receiving {} but expecting euclidean, manhattan, cosine, or minkowski".format(
                        metric
                    )
                )
            distances_.append(dist)
            neigh_class.append(train_class)

        # Sort distances
        sorted_ind = np.argsort(distances_, axis=0)  # by rows

        # Identify k nearest neighbours
        sorted_dist = [distances_[i] for i in sorted_ind][:k]
        neighbours_class = [neigh_class[i] for i in sorted_ind][:k]

        return neighbours_class, sorted_dist

    def get_neighbours_for_all(X_train, X_test, Y, k, p, metric):
        kclusters = []
        distances = []
        # Loop over rows in test set
        for test_row in X_test:
            nearest_neighbours, n_dist = get_neighbours(
                X_train, test_row, Y, k, p, metric
            )
            kclusters.append(nearest_neighbours)
            distances.append(n_dist)
        return np.array(kclusters), np.array(distances)

    def knn_predict(X_train, X_test, Y, k, p, metric):
        neighbours_class, cluster_distances = get_neighbours_for_all(
            X_train, X_test, Y, k, p, metric
        )
        neighbours_class = neighbours_class[
            :, 1:
        ]  # removing the closest neighbor which is the point itself
        print(
            "The nearest neighbor clusters distances are: \n{}".format(
                cluster_distances[:5, :]
            )
        )
        print(
            "The nearest neighbor clusters predictions: \n{}".format(
                neighbours_class[:5, :]
            )
        )

        # Finding the dominant class in each row of neighbor classes
        cluster_predictions, counts = mode(neighbours_class, axis=1)
        print(cluster_predictions)
        print(y_test_subset[0], cluster_predictions[0])

        # showing accuracy and f1_scores
        models_accuracy = accuracy_score(y_test_subset, cluster_predictions)
        f1_scores = f1_score(y_test_subset, cluster_predictions, average="macro")
        print(models_accuracy, f1_scores)

        return cluster_predictions

    # select options for model
    model_options = ['Logistic Regression', 'Stochastic Gradient Descent with Logistic Regression', 'KNN']
    model_select = st.selectbox(
        label='Select model for prediction',
        options=model_options,
    )
    st.write('You selected : {}'.format(
        model_select))

    # ----------------------------------------------
    # Logistic Regression
    # ----------------------------------------------

    if model_options[0]  == model_select:
        st.header("Logistic Regression")
        lr_value = st.slider(
            "Select a learning rate", min_value=0.01, max_value=0.1, step=0.01
        )
        st.write("Logistic Regression Learning Rate:", lr_value)

        num_of_iter = st.slider(
            "Select number of iterations", min_value=100, max_value=1000, step=100
        )
        st.write("Logistic Regression Number of Iterations:", num_of_iter)
        if st.button("Train Model"):
            accuracies = {}
            logreg_model = LogisticRegression_jamie()
            logreg_model.fit(X_train, y_train, alpha=lr_value, iter=num_of_iter)
            Y_pred = logreg_model.predict(X_test)
            predictions = [round(value) for value in Y_pred]

            # evaluate predictions
            accuracy = accuracy_score(y_test, predictions)
            accuracies["Logistic Regression"] = accuracy * 100.0
            st.write("Accuracy: %.2f%%" % (accuracy * 100.0))

            st.session_state.model = ("logreg",logreg_model)

    elif model_options[1]  == model_select:
        st.header("Stochastic Gradient Descent with Logistic Regression")
        # Number of iterations: maximum iterations to run the iterative SGD
        sdg_num_iterations = st.number_input(
            label="Enter the number of maximum iterations on training data",
            min_value=1,
            max_value=5000,
            value=500,
            step=100,
            key="sgd_num_iterations_numberinput",
        )
        st.write("maximum iterations to: {}".format(sdg_num_iterations))

        # learning_rate: Constant that multiplies the regularization term. Ranges from [0 Inf)
        sdg_learning_rate = st.text_input(
            label="Input one alpha value",
            value="0.001",
            key="sdg_learning_rate_numberinput",
        )
        sdg_learning_rate = float(sdg_learning_rate)
        st.write("learning rate: {}".format(sdg_learning_rate))

        # tolerance: stopping criteria for iterations
        sgd_batch_size = st.text_input(
            label="Input a batch size value", value="50", key="sgd_batch_size_textinput"
        )
        sgd_batch_size = int(sgd_batch_size)
        st.write("batch_size: {}".format(sgd_batch_size))

        sgd_params = {
            "num_iterations": sdg_num_iterations,
            "batch_size": sgd_batch_size,
            "learning_rate": sdg_learning_rate,
        }
        if st.button("Train Model"):
            try:
                X_train = pd.DataFrame(X_train.toarray())
                X_test = pd.DataFrame(X_test.toarray())
                sgd = StochasticLogisticRegression(
                    num_iterations=sgd_params["num_iterations"],
                    learning_rate=sgd_params["learning_rate"],
                    batch_size=sgd_params["batch_size"],
                )
                sgd.fit(X_train, np.ravel(y_train))
                st.session_state[model_options[1]] = sgd
                y_pred_sgd = sgd.predict(X_test)
                st.session_state.model = ("sgd",sgd)
                accuracy = accuracy_score(y_test, y_pred_sgd)
                st.write(f"Accuracy: {accuracy * 100.0}%")
            except ValueError as err:
                st.write({str(err)})

    # ----------------------------------------------
    # KNN
    # ----------------------------------------------

    elif model_options[2]  == model_select:
        st.header("KNN")
        # Hyper-parameters
        num_neighhbors = st.slider(
            "Select a learning rate", min_value=3, max_value=20, step=1
        )
        st.write("Choose number of neighbors:", num_neighhbors)

        metric_options = ["manhattan", "minkowski", "euclidean", "cosine"]
        metric = st.selectbox(
            label="Select metric for distance calculation",
            options=metric_options,
            index=1,
        )
        st.write("You selected : {}".format(metric))

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

        cluster_predictions = knn_predict(
            X_train_subset.to_numpy(),
            X_test_subset.to_numpy(),
            y_train_subset,
            num_neighhbors,
            p,
            metric,
        )

        st.write(X_test_subset.to_numpy()[0].shape)
        st.write("cluster_predictions", cluster_predictions)
        st.session_state.model = ("knn", (X_train_subset.to_numpy(),
                    y_train_subset,
                    num_neighhbors,
                    p,
                    metric,))

    st.subheader("Predict MBTI")
    form = st.form(key="user_form")
    user_input = form.text_input('Enter the post content')

    if form.form_submit_button('Predict'):
        if 'model' not in st.session_state:
            st.error("Please select a model to train before predicting")
        if user_input:
            temp_dp = pd.DataFrame([user_input],columns = ["posts"])
            # user_input = vetorize_posts(temp_df["posts"])
            user_input = vect.transform(temp_dp["posts"])
            modelname, model = st.session_state.model
            if modelname == "logreg":
                try:
                    result = model.predict(user_input)
                    st.success(f"Prediction: {result}")
                except Exception as e:
                    st.error(e) 
            elif modelname == "sgd":
                try:
                    # user_input = pd.DataFrame(X_train.toarray())
                    result = model.predict(user_input)
                    st.success(f"Prediction: {result}")
                except Exception as e:
                    st.error("Invalid input") 
            else:
                X_train_subset,y_train_subset,num_neighhbors,p,metric = model
                user_input = pd.DataFrame(user_input.toarray())
                # st.write(user_input.to_numpy().flatten().shape)
                # cluster_predictions = knn_predict(
                #     X_train_subset,
                #     user_input.to_numpy().flatten(),
                #     y_train_subset,
                #     num_neighhbors,
                #     p,
                #     metric,
                # )
                # st.success("new_post_cluster_predictions", cluster_predictions)

        else:
            st.error("Please enter some input before predicting.")

