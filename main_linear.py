import matplotlib.pyplot as plt
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

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(california_housing_X_train, california_housing_Y_train)

# Make predictions using the testing set
california_housing_Y_pred = reg.predict(california_housing_X_test)

# The coefficients
print("\nCoefficients: \n", reg.coef_)
# Intercept
print("Intercept: \n", reg.intercept_)
# Mean square error
print("Mean square error: %.6f" % mean_squared_error(california_housing_Y_test, california_housing_Y_pred))
# Explained variance score: 1 is perfect prediction
print("Variance score: %.6f" % r2_score(california_housing_Y_test, california_housing_Y_pred))

plt.scatter(california_housing_X_test, california_housing_Y_test, color='blue')
plt.plot(california_housing_X_test, california_housing_Y_pred, color='red', linewidth=3)

plt.xticks()
plt.yticks()

plt.show()
