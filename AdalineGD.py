import numpy as np

class AdalineGD:
    """
    ADAptive LInear NEuron Classifier

    Params:
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Number of passes over the training set (epochs)
    random_state: int
        Seed variable for random weight initialization
    
    Attributes:
    w_: 1-d array
        weights of the perceptron after fitting
    b_: scalar
        bias of the perceptron after fitting
    losses_: list
        number of errors in each epoch during fitting

    """

    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Params:
        X: {array-like} (matrix), shape = [n_examples, n_features]
            A matrix consisting of training vectors. n_examples
            is the number of examples and n_features is the number
            of features.
        y: array-like, shape = [n_examples]
            Target values

        Returns:
        self
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(0.0, 0.01, X.shape[1])
        self.b_ = 0.0
        self.losses_ = []

        for _ in range(self.n_iter):
            net_i = self.net_input(X)
            out = self.activation(net_i)
            errors = y - out
            self.w_ += 2 * self.eta * X.T.dot(errors) / X.shape[0]
            self.b_ += 2 * self.eta * sum(errors) / X.shape[0]
            self.losses_.append(sum(errors ** 2)/X.shape[0])

        return self
        
    def net_input(self, X):
        """Returns an array of the net inputs for each input in X (weights * inputs + bias)"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, z):
        """Returns result of passing net input through an activation function (here, identity)"""
        return z
    
    def predict(self, X):
        """Return an array of predicted classes"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, 0)

