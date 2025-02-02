import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic dataset for regression
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + 2 * (X**2) + np.random.randn(100, 1)  # Quadratic relation y = 4 + 3x + 2x² + noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return rmse_train, rmse_test, r2_train, r2_test

# Train Decision Tree Regression
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
tree_rmse_train, tree_rmse_test, tree_r2_train, tree_r2_test = evaluate_model(tree_reg, X_train, y_train, X_test, y_test)

# Train Random Forest Regression
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)
rf_rmse_train, rf_rmse_test, rf_r2_train, rf_r2_test = evaluate_model(rf_reg, X_train, y_train, X_test, y_test)

# Train Lasso Regression
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)
lasso_rmse_train, lasso_rmse_test, lasso_r2_train, lasso_r2_test = evaluate_model(lasso_reg, X_train, y_train, X_test, y_test)

# Train Logistic Regression (for classification)
# For Logistic Regression, let's generate a synthetic classification dataset
# Train Logistic Regression (for classification)
# For Logistic Regression, let's generate a synthetic classification dataset
from sklearn.datasets import make_classification
X_class, y_class = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train_class, y_train_class)
log_rmse_train = np.sqrt(mean_squared_error(y_train_class, log_reg.predict(X_train_class)))
log_rmse_test = np.sqrt(mean_squared_error(y_test_class, log_reg.predict(X_test_class)))
log_r2_train = r2_score(y_train_class, log_reg.predict(X_train_class))
log_r2_test = r2_score(y_test_class, log_reg.predict(X_test_class))

# Print Results
print(f"Decision Tree         -> Train RMSE: {tree_rmse_train:.2f}, Test RMSE: {tree_rmse_test:.2f}, Train R²: {tree_r2_train:.2f}, Test R²: {tree_r2_test:.2f}")
print(f"Random Forest         -> Train RMSE: {rf_rmse_train:.2f}, Test RMSE: {rf_rmse_test:.2f}, Train R²: {rf_r2_train:.2f}, Test R²: {rf_r2_test:.2f}")
print(f"Lasso Regression      -> Train RMSE: {lasso_rmse_train:.2f}, Test RMSE: {lasso_rmse_test:.2f}, Train R²: {lasso_r2_train:.2f}, Test R²: {lasso_r2_test:.2f}")
print(f"Logistic Regression   -> Train RMSE: {log_rmse_train:.2f}, Test RMSE: {log_rmse_test:.2f}, Train R²: {log_r2_train:.2f}, Test R²: {log_r2_test:.2f}")


# Visualization for Regression models
X_sorted = np.sort(X, axis=0)
plt.scatter(X_train, y_train, label="Training Data", color="blue", alpha=0.6)
plt.scatter(X_test, y_test, label="Testing Data", color="red", alpha=0.6)
plt.plot(X_sorted, tree_reg.predict(X_sorted), label="Decision Tree", linestyle="dashed", color="green")
plt.plot(X_sorted, rf_reg.predict(X_sorted), label="Random Forest", color="black")
plt.plot(X_sorted, lasso_reg.predict(X_sorted), label="Lasso", linestyle="dotted", color="purple")
plt.legend()
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.title("Comparison of Regression Models")
plt.show()
