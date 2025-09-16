#Necessary imports

from features import preprocessing
import numpy as np
from data import X, y

# training Linear Regression Model
X_prepared = prepare_features(X.values) # .values converts pandas to numpy
y_prepared = y.values

# 2. Initialize parameters
alpha = 0.01
iterations = 150
initial_theta = np.zeros(X_prepared.shape[1])
final_theta, cost_history = gradient_descent(X, y, initial_theta, alpha, iterations)