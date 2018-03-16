from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split

# import some and load data to play with
digits = datasets.load_digits()

# Split and shuffle the data
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.20, random_state=42)
# offset = int(X.shape[0] * 0.8)
# X_train, y_train = X[:offset], y[:offset]
# X_test, y_test = X[offset:], y[offset:]

# Create the model
logreg = linear_model.LogisticRegression(C=0.01)

# We fit the data
logreg.fit(X_train, y_train)

# Print the results
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

digits_y_pred = logreg.predict(X_test)
print("Test set y:")
print(y_test)
print("Prediction set y:")
print(digits_y_pred)
print("Absolute value of Y test and Y prediction")
print(abs(y_test - digits_y_pred))
