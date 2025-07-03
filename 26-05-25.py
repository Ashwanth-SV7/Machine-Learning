import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('auto-mpg.csv')
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df.dropna(subset=['mpg', 'horsepower'], inplace=True)

X = df[['horsepower']].values
y = df['mpg'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

theta = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train
y_pred = X_test_b @ theta

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

print("MSE:", mse(y_test, y_pred))
print("RMSE:", rmse(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
