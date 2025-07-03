import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

X_b = np.c_[np.ones((X.shape[0], 1)), X]  

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Intercept (theta_0):", theta_best[0])
print("Slope (theta_1):", theta_best[1])

y_pred = X_b.dot(theta_best)

plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X, y_pred, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression using NumPy')
plt.legend()
plt.grid(True)
plt.show()
