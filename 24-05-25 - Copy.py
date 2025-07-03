# Import necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseValue'] = housing.target  # Add target column

# Check correlation to select most relevant features
correlation = df.corr()
print("Correlation with target:\n", correlation['MedHouseValue'].sort_values(ascending=False))

# Select the 4 most relevant features based on correlation
# We'll avoid 'MedHouseValue' itself and pick the top 4
top_features = correlation['MedHouseValue'].drop('MedHouseValue').sort_values(ascending=False).head(4).index.tolist()
print("\nTop 4 selected features:", top_features)

# Define feature matrix X and target vector y
X = df[top_features]
y = df['MedHouseValue']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\nModel Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
