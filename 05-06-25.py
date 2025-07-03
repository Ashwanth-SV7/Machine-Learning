import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Load the Wine dataset
data = load_wine()
X, y = data.data, data.target

# Shuffle the dataset
X, y = shuffle(X, y, random_state=42)

# Number of folds
K = 5
fold_size = len(X) // K
accuracies = []

for fold in range(K):
    # Create test indices for current fold
    start = fold * fold_size
    end = start + fold_size if fold != K - 1 else len(X)

    # Split into training and testing sets
    X_test = X[start:end]
    y_test = y[start:end]

    X_train = np.concatenate((X[:start], X[end:]), axis=0)
    y_train = np.concatenate((y[:start], y[end:]), axis=0)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")

# Final Results
print("\nAll Fold Accuracies:", [f"{a:.4f}" for a in accuracies])
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
