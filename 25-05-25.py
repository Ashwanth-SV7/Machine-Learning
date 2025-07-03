import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('auto-mpg.csv')
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df.dropna(subset=['mpg', 'horsepower'], inplace=True)

X = df[['horsepower']].values
y = df['mpg'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_s = scaler.transform(X_plot)

degrees = [1, 2, 3]
plt.figure(figsize=(10, 6))

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train_s)
    X_test_poly = poly.transform(X_test_s)
    X_plot_poly = poly.transform(X_plot_s)

    model = LinearRegression().fit(X_train_poly, y_train)
    y_test_pred = model.predict(X_test_poly)
    y_plot = model.predict(X_plot_poly)
  
    if d == 1:
        plt.scatter(X_train, y_train, color='blue', alpha=0.3, label='Train')
        plt.scatter(X_test, y_test, color='red', alpha=0.3, label='Test')
    plt.plot(X_plot, y_plot, label=f'Degree {d} (R²={r2_score(y_test, y_test_pred):.2f})')

    print(f"Degree {d}: R² = {r2_score(y_test, y_test_pred):.3f}, MSE = {mean_squared_error(y_test, y_test_pred):.2f}")

plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('MPG Prediction using Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
