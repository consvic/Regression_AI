import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the california housing dataset
california_housing = datasets.fetch_california_housing()
X = california_housing.data
Y = california_housing.target
# Use only one feature
# california_housing_X = california_housing.data[:, np.newaxis, 7]
california_housing_X = california_housing.data
offset = int(X.shape[0] * 0.8)

# Split the data into training/testing sets
california_housing_X_train = california_housing_X[:offset]
california_housing_X_test = california_housing_X[offset:]

# Split the targets into training/testing sets
california_housing_Y_train = Y[:offset]
california_housing_Y_test = Y[offset:]

# alpha value
alpha_in = 100000
# alpha values for graph
n_alphas = 200
alphas = np.logspace(-3, 7, n_alphas)

# Ridge regression and train model
reg = linear_model.Ridge(alpha=alpha_in).fit(california_housing_X_train, california_housing_Y_train)
coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(california_housing_X_train, california_housing_Y_train)
    coefs.append(ridge.coef_)

# Plot Ridge coefficients
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
