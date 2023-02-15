import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape # get number of samples and features based on dimension of the input
        self.weights = np.zeros(self.features)
        self.bias = 0

        for _ in range(self.n_iters):
            pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(pred)

            # calculate each step of the gradient descent by differentiating the weight and bias
            dw = (1/n_samples) * np.dot(X.T, (predictions - y)) #
            db = (1/n_samples) * np.sum(predictions - y)

            # update the new weights and bias
            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db


    def predict(self, X):
        """
        :param X: input value of the test data
        :return: a list of predicted output using Logistic Regression
        """
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred

