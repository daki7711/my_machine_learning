from sklearn.utils import shuffle
from numpy import random, dot


class MachineLearningLR:

    def __init__(self):
        pass

    # function that returns a TRUE value if the error difference is smaller than a certain error rate
    def error_difference(self, before_error, current_error):
        if before_error == 0:
            return False
        error_rate = 0.001
        return abs(current_error - before_error) / before_error <= error_rate

    # function to compute hypothesis / predictions
    def predictions(self, X, w):
        return dot(X, w)

    # function to compute gradient of error function w.r.t. weight w
    def gradient(self, X, y, w):
        grad = dot(X.T, (self.predictions(X, w) - y))
        return grad

    # function to compute the error for current values of weight w
    def R(self, X, y, w):
        return (1 / len(y)) * ((self.predictions(X, w) - y) ** 2).sum()

    def online_update(self, X, y, w, alpha):
        """
        One epoch of stochastic gradient descent (i.e. one sweep of the dataset).

        Parameters
        ----------
        X : NumPy array of features (size : no of examples X features)
        y : Numpy array of class labels (size : no of examples X 1)
        w : array of coefficients from the previous iteration

        Returns
        -------
        Coefficients of the model (after updating)
        """
        error = 0

        for i in range(len(y)):
            w = w - alpha * (dot(w, X[i]) - y[i]) * X[i]
            error += self.R(X, y, w)

        error /= len(y)

        return error, w

    def batch_update(self, X, y, w, alpha):
        """
        One iteration of full-batch gradient descent.

        Parameters
        ----------
        X : NumPy array of features (size : no of examples X features)
        y : Numpy array of class labels (size : no of examples X 1)
        w : array of coefficients from the previous iteration

        Returns
        -------
        Coefficients of the model (after updating)
        """
        w = w - alpha * (1 / len(y)) * self.gradient(X, y, w)

        return self.R(X, y, w), w

    def mini_batch_update(self, X, y, w, alpha, batch_size):
        """
        One epoch of mini-batch SGD over the entire dataset (i.e. one sweep of the dataset).

        Parameters
        ----------
        X : NumPy array of features (size : no of examples X features)
        y : Numpy array of class labels (size : no of examples X 1)
        w : array of coefficients from the previous iteration
        batch_size : size of the batch for gradient update

        Returns
        -------
        Coefficients of the model (after updating)
        """
        mini_batch_error = 0
        for i in range(0, len(y), batch_size):
            if i + batch_size < len(y):
                w = w - alpha * (1 / batch_size) * self.gradient(X[i:i + batch_size], y[i:i + batch_size], w)
                error = self.R(X[i:i + batch_size], y[i:i + batch_size], w)
            else:
                w = w - alpha * (1 / batch_size) * self.gradient(X[i:], y[i:], w)
                error = self.R(X[i:], y[i:], w)
            mini_batch_error += error
        return mini_batch_error / (len(y) / batch_size), w

    def least_squares_grad_desc(self, X, y, maxIter, alpha, update, *batch_size):
        """
        Implements least squares with gradient descent.

        Parameters
        ----------
        X : NumPy array of features (size : no of examples X features)
        y : Numpy array of class labels (size : no of examples X 1)
        maxIter : Maximum number of iterations allowed
        alpha : Learning rate
        update : update function to utilize (one of online, batch, mini-batch)
        batch_size : number of examples in a batch (only useful when update = mini_batch_update)

        Returns
        -------
        The entire list of errors in each epoch
        Coefficients of the model (after updating)

        Note : *batch_size is an optional argument and only to be used when doing mini-batch Gradient Descent
        """
        error_list = [float('inf')]
        w = random.rand(len(X[0]))
        for i in range(maxIter):
            X, y = shuffle(X, y)
            if update == self.mini_batch_update:
                error, w = update(X, y, w, alpha, batch_size[0])
            else:
                error, w = update(X, y, w, alpha)

            error_list.append(error)
            if i > 2 and self.error_difference(error_list[-2], error_list[-1]):
                return error_list, w
        return error_list, w
