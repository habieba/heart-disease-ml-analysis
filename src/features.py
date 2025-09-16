# Imports
import numpy as np

#Linear Regression Code

#hypothesis function
def predict(X, theta):
    """
      Calculates the hypothesis for linear regression.

      Args:
        X (np.array): The feature matrix.
        theta (np.array): The parameter vector.

      Returns:
        np.array: The predicted values.
      """
    return X @ theta

#cost function
def cost(X, y, theta):
    """
      Calculates the Mean Squared Error cost.

      Args:
        X (np.array): The feature matrix.
        y (np.array): The actual target values.
        theta (np.array): The parameter vector.

      Returns:
        float: The cost.
      """
    m = len(y)  # Number of training examples

    # Calculate predictions (using our predict function)
    predictions = predict(X, theta)

    errors = predictions - y
    # This dot product is equivalent to np.sum(errors**2)
    sum_squared_errors = errors @ errors

    return (1 / (2 * m)) * sum_squared_errors

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = [] # useful for confirming if gradient descent is working
    for _ in range(iterations):
        errors = predict(X, theta) - y
        gradient = (X.T @ errors) / m
        # Update theta using the rule
        theta = theta - alpha * gradient
        cost_history.append(theta)

    return theta, cost_history

def preprocessing(X):
    """
      Performs feature scaling and adds a bias term.

      Args:
        X (np.array): The original feature matrix.

      Returns:
        np.array: The prepared feature matrix.
      """
    # 1. Scale the original features
    mean = np.mean(X, 0)
    std = np.std(X, 0)
    X_scaled = (X - mean) / std

    # 2. Add the bias term (column of ones)
    X_prepared = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

    return X_prepared