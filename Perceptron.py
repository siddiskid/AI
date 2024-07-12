import numpy as np

class Perceptron:
    """
    Perceptron classifier
    
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
    errors: list
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
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Returns an array of the net inputs (inputs multiplied by weights + biases)"""
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        """Return an array of predicted classes"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)