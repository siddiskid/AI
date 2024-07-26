import numpy as np

class AdalineSGD:
    """
    ADAptive LInear NEuron Classifier that learns using Stochastic Gradient Descent

    Params:
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Number of passes over the training set (epochs)
    shuffle: bool
        If True, shuffles data every epoch to prevent cycles
    random_state: int
        Seed variable for random weight initialization
    
    Attributes:
    w_: 1-d array
        weights of the perceptron after fitting
    b_: scalar
        bias of the perceptron after fitting
    losses_: list
        loss averaged over all training examples in each epoch

    """

    def __init__(self, eta = 0.01, n_iter = 50, shuffle = True, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.initialized_w = False
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
        self.initialize_weights(X.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self.shuffle_weights(X, y)
            epoch_losses = []
            for xi, target in zip(X, y):
                epoch_losses.append(self.update_weights(xi, target))
            self.losses_.append(np.mean(epoch_losses))
        return self
    
    def online_fit(self, X, y):
        """Fit data but only update the weights, not reinitialize"""
        if not self.initialized_w:
            self.initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self.update_weights(xi, target)
        else:
            self.update_weights(X, y)
        # no need to check just zip?
        # for xi, target in zip(X, y):
        #         self.update_weights(xi, target)
        return self
    
    def update_weights(self, xi, target):
        """Update Adaline weights based on a single training example, returns loss"""
        error = target - self.activation(self.net_input(xi))
        self.w_ += self.eta * error * xi
        self.b_ += self.eta * error
        return error ** 2

    def shuffle_weights(self, X, y):
        """Randomly shuffle """
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def initialize_weights(self, size):
        """Initialize weights and bias to small, random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(0.0, 0.01, size)
        self.b_ = 0.0
        self.initialized_w = True
        
    def net_input(self, X):
        """Returns an array of the net inputs for each input in X (weights * inputs + bias)"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, z):
        """Returns result of passing net input through an activation function (here, identity)"""
        return z
    
    def predict(self, X):
        """Return an array of predicted classes"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

