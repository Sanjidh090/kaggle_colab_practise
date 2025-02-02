import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic dataset
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Model Evaluation
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {rmse_train:.2f}, Test RMSE: {rmse_test:.2f}")
print(f"Train R²: {r2_train:.2f}, Test R²: {r2_test:.2f}")

# Plot Model Fit
plt.scatter(X_train, y_train, label="Training Data", color="blue", alpha=0.6)
plt.scatter(X_test, y_test, label="Testing Data", color="red", alpha=0.6)
plt.plot(X, model.predict(X), color="black", linewidth=2, label="Model Prediction")
plt.legend()
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.title("Linear Regression Model Fit")
plt.show()

# Residual Analysis
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, color="purple", alpha=0.6)
plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
