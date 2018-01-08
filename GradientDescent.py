import numpy as np


class GradientDescent(object):
    """Preform the gradient descent optimization algorithm for an arbitrary
    cost function.
    """

    def __init__(self, cost, gradient, predict_func, fit_intercept=True,
                 alpha=0.01,
                 num_iterations=10000):
        """Initialize the instance attributes of a GradientDescent object.

        Parameters
        ----------
        cost: The cost function to be minimized.
        gradient: The gradient of the cost function.
        predict_func: A function to make predictions after the optimization has
            converged.
        alpha: The learning rate.
        num_iterations: Number of iterations to use in the descent.

        Returns
        -------
        self: The initialized GradientDescent object.
        """
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.num_iterations = num_iterations


    def fit(self, X, y):
        """Run the gradient descent algorithm for num_iterations repetitions.

        Parameters
        ----------
        X: A two dimenstional numpy array.  The training data for the
            optimization.
        y: A one dimenstional numpy array.  The training response for the
            optimization.

        Returns
        -------
        self:  The fit GradientDescent object.
        """
        if self.fit_intercept:
            X = np.concatenate((X, np.ones(X.shape[0]).reshape(X.shape[0], 1)), axis=1)
        self.coeffs = np.zeros(X.shape[1])
        for i in range(self.num_iterations):
            self.coeffs -= self.alpha * self.gradient(X, y, self.coeffs)

    def predict(self, X):
        """Call self.predict_func to return predictions.

        Parameters
        ----------
        X: Data to make predictions on.

        Returns
        -------
        preds: A one dimensional numpy array of predictions.
        """
        if self.fit_intercept:
            X = np.concatenate((X, np.ones(X.shape[0]).reshape(X.shape[0], 1)), axis=1)
        return self.predict_func(X, self.coeffs)
