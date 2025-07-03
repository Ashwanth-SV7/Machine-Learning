import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target
classes = digits.target_names

# Binarize output for ROC-AUC (One-vs-Rest)
y_bin = label_binarize(y, classes=classes)
n_classes = y_bin.shape[1]

# Train-test split
X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    X, y, y_bin, test_size=0.2, random_state=42
)

# Train SVM classifier
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)
y_proba = svm.predict_proba(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# ROC-AUC (One-vs-Rest)
roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr')

# Display results
print("✅ Accuracy:", round(acc, 3))
print("\n📊 Classification Report:\n", report)
print("🎯 ROC-AUC (OvR):", round(roc_auc, 3))

# Confusion Matrix Plot
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - SVM on Digits Dataset")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
