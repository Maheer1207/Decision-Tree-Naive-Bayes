"""
For the following:
- N: Number of samples.
- D: Dimension of input features.
- C: Number of classes (labels). We assume the class starts from 0.
"""

import numpy as np


class Params:
    def __init__(self, means, covariances, priors, num_features, num_classes):
        """ This class represents the parameters of the Naive Bayes model,
            where the generative model is modeled as a Gaussian.
        NOTE: We assume lables are 0 to K - 1, where K is number of classes.

        We have three parameters to keep track of:
        - self.means (ndarray (shape: (K, D))): Mean for each of K Gaussian likelihoods.
        - self.covariances (ndarray (shape: (K, D, D))): Covariance for each of K Gaussian likelihoods.
        - self.priors (shape: (K, 1))): Prior probabilty of drawing samples from each of K class.

        Args:
        - num_features (int): The number of features in the input vector
        - num_classes (int): The number of classes in the task.
        """

        self.D = num_features
        self.C = num_classes

        # Shape: K x D
        self.means = means

        # Shape: K x D x D
        self.covariances = covariances

        # Shape: K x 1
        self.priors = priors

        assert self.means.shape == (self.C, self.D), f"means shape mismatch. Expected: {(self.C, self.D)}. Got: {self.means.shape}"
        assert self.covariances.shape == (self.C, self.D, self.D), f"covariances shape mismatch. Expected: {(self.C, self.D, self.D)}. Got: {self.covariances.shape}"
        assert self.priors.shape == (self.C, 1), f"priors shape mismatch. Expected: {(self.C, 1)}. Got: {self.priors.shape}"


def train_nb(train_X, train_y, num_classes, **kwargs):
    """ This trains the parameters of the NB model, given training data.

    Args:
    - train_X (ndarray (shape: (N, D))): NxD matrix storing N D-dimensional training inputs.
    - train_y (ndarray (shape: (N, 1))): Column vector with N scalar training outputs (labels).

    Output:
    - params (Params): The parameters of the NB model.
    """
    assert len(train_X.shape) == len(train_y.shape) == 2, f"Input/output pairs must be 2D-arrays. train_X: {train_X.shape}, train_y: {train_y.shape}"
    (N, D) = train_X.shape
    assert train_y.shape[1] == 1, f"train_Y must be a column vector. Got: {train_y.shape}"

    # Shape: C x D
    means = np.zeros((num_classes, D))

    # Shape: C x D x D
    covariances = np.tile(np.eye(D), reps=(num_classes, 1, 1))

    # Shape: C x 1
    priors = np.ones(shape=(num_classes, 1)) / num_classes

    # Calculate means, covariances, and priors for each class
    for i in range(num_classes):
        # Output to class/label i, and stores either True or False
        class_indices = (train_y == i).flatten()
        # Selects row of class_data for the class indices that matches i
        class_data = train_X[class_indices, :]
        # Uses the mean and cov function for mean and covariance of class_data
        # Keep the rowvar=False to indicate each col repr a var while rows are observations
        means[i, :] = np.mean(class_data, axis=0)
        covariances[i, :, :] = np.cov(class_data, rowvar=False)
        # Here we compute the prior prob
        priors[i, 0] = class_data.shape[0] / N


    params = Params(means, covariances, priors, D, num_classes)
    return params


def predict_nb(params, X):
    """ This function predicts the probability of labels given X.

    Args:
    - params (Params): The parameters of the NB model.
    - X (ndarray (shape: (N, D))): NxD matrix with N D-dimensional inputs.

    Output:
    - probs (ndarray (shape: (N, K))): NxK matrix storing N K-vectors (i.e. the K class probabilities)
    """
    assert len(X.shape) == 2, f"Input/output pairs must be 2D-arrays. X: {X.shape}"
    (N, D) = X.shape

    probs = np.zeros((N, params.C))
    unnormalized_probs = np.zeros((N, params.C))
    
    
    for i in range(params.C):
        means = params.means[i, :]
        covariances = params.covariances[i, :, :]
        priors = params.priors[i, 0]
        # Gets the deviation from mean
        dev = X - means
        # Calculates the Gaussian likelihood for each input samples
        exponent = -0.5 * np.sum(np.dot(dev, np.linalg.inv(covariances)) * dev, axis=1)
        class_probs = priors * np.exp(exponent)
        # Stores the unnormalized prob of the current class
        unnormalized_probs[:, i] = class_probs

    # Check if the sum is zero before normalization
    sum_unnormalized_probs = np.sum(unnormalized_probs, axis=1, keepdims=True)
    # Avoid division by zero by setting zero-sum rows to 1 (probs will be zero)
    sum_unnormalized_probs[sum_unnormalized_probs == 0] = 1
    # Normalize the probabilities
    probs = unnormalized_probs / sum_unnormalized_probs

    return probs
