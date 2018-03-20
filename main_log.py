from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split

# import some and load data to play with
digits = datasets.load_digits()

# Split and shuffle the data
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.20, random_state=42)

# Create the model wtih C=1
logreg = linear_model.LogisticRegression(C=1)

# We fit the data
logreg.fit(X_train, y_train)

# Print description of the dataset
print(digits.DESCR)
print("This dataset is made up of 1797 8x8 images. Each image, like the one shown below, is of a hand-written digit.\nIn order to utilize an 8x8 figure like this, we’d have to first transform it into a feature vector with length 64.")
print("Target names:")
print(digits.target_names)

# Print the results as well as test and prediction arrays
print("C=1")
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

digits_y_pred = logreg.predict(X_test)
print("Test set y:")
print(y_test)
print("Prediction set y:")
print(digits_y_pred)
print("Absolute value of Y test and Y prediction")
print(abs(y_test - digits_y_pred))

# Create the model wtih C=100 and fit the data
logreg = linear_model.LogisticRegression(C=100)
logreg.fit(X_train, y_train)
digits_y_pred2 = logreg.predict(X_test)

# Print the results as well as test and prediction arrays
print("C=100")
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

# Create the model wtih C=0.01 and fit the data
logreg = linear_model.LogisticRegression(C=0.01)
logreg.fit(X_train, y_train)
digits_y_pred3 = logreg.predict(X_test)

# Print the results as well as test and prediction arrays
print("C=0.01")
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))
