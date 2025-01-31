import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.polynomial.polynomial import Polynomial
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

results = pd.read_csv('intensity.csv')

x_liquid = results['distance_liquid']
ln_intensity = results['-ln_intensity']

# Assuming x_data and y_data are your data points
x_train, x_val, y_train, y_val = train_test_split(x_liquid, ln_intensity, test_size=0.2, random_state=42)


plt.scatter(x_train, y_train, s=5, label='training data')
plt.scatter(x_val, y_val, s=5, label='validation data')
plt.legend()
plt.show()


# Degrees to test
degrees = range(1, 10)
train_errors = []
val_errors = []

# Regularization strength
alpha = 1.0  # You can adjust this value

for d in degrees:
    # Create a pipeline: PolynomialFeatures + Ridge Regression
    model = make_pipeline(PolynomialFeatures(d), Lasso(alpha=alpha))
    
    # Fit model on training data
    model.fit(x_train.values.reshape(-1, 1), y_train)
    
    # Predict on training and validation data
    y_train_pred = model.predict(x_train.values.reshape(-1, 1))
    y_val_pred = model.predict(x_val.values.reshape(-1, 1))
    
    # Calculate MSE
    train_errors.append(np.mean((y_train - y_train_pred) ** 2))
    val_errors.append(np.mean((y_val - y_val_pred) ** 2))

# Plot the results
plt.plot(degrees, train_errors, label="Training Error")
plt.plot(degrees, val_errors, label="Validation Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title(f"Polynomial Fitting with L2 Regularization (alpha={alpha})")
plt.legend()
plt.show()
train_errors = []
val_errors = []
alphas = range(0,50)
for a in alphas:
    # Create a pipeline: PolynomialFeatures + Ridge Regression
    model = make_pipeline(PolynomialFeatures(10), Lasso(alpha=a))
    
    # Fit model on training data
    model.fit(x_train.values.reshape(-1, 1), y_train)
    
    # Predict on training and validation data
    y_train_pred = model.predict(x_train.values.reshape(-1, 1))
    y_val_pred = model.predict(x_val.values.reshape(-1, 1))
    
    # Calculate MSE
    train_errors.append(np.mean((y_train - y_train_pred) ** 2))
    val_errors.append(np.mean((y_val - y_val_pred) ** 2))

# Plot the results
plt.plot(alphas, train_errors, label="Training Error")
plt.plot(alphas, val_errors, label="Validation Error")
plt.xlabel("Lambda")
plt.ylabel("Mean Squared Error")
plt.title(f"Polynomial Fitting with L2 Regularization (degree=10)")
plt.legend()
plt.show()
