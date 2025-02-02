import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic dataset (non-linear)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + 2 * (X**2) + np.random.randn(100, 1)  # Quadratic relation y = 4 + 3x + 2x² + noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Transform data into polynomial features (degree 2)
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Train Polynomial Regression Model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predictions
y_train_pred_poly = poly_model.predict(X_train_poly)
y_test_pred_poly = poly_model.predict(X_test_poly)
# Compute MSE & R²
mse_train_poly = mean_squared_error(y_train, y_train_pred_poly)
mse_test_poly = mean_squared_error(y_test, y_test_pred_poly)

rmse_train_poly = np.sqrt(mse_train_poly)
rmse_test_poly = np.sqrt(mse_test_poly)

r2_train_poly = r2_score(y_train, y_train_pred_poly)
r2_test_poly = r2_score(y_test, y_test_pred_poly)

print(f"Polynomial Regression Train RMSE: {rmse_train_poly:.2f}, Test RMSE: {rmse_test_poly:.2f}")
print(f"Polynomial Regression Train R²: {r2_train_poly:.2f}, Test R²: {r2_test_poly:.2f}")
# Sort values for smooth plotting
X_sorted = np.sort(X, axis=0)
y_poly_pred_sorted = poly_model.predict(poly_features.transform(X_sorted))

plt.scatter(X_train, y_train, label="Training Data", color="blue", alpha=0.6)
plt.scatter(X_test, y_test, label="Testing Data", color="red", alpha=0.6)
plt.plot(X_sorted, y_poly_pred_sorted, color="black", linewidth=2, label="Polynomial Fit")
plt.legend()
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.title("Polynomial Regression Model Fit (Degree = 2)")
plt.show()
