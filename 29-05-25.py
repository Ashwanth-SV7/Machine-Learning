import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("/mnt/data/titanic.csv")

# Basic Cleaning
# Drop rows with missing Age or Embarked
data = data.dropna(subset=['Age', 'Embarked'])

# Fill missing Fare with median
data['Fare'].fillna(data['Fare'].median(), inplace=True)

# Encode categorical features
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])  # male=1, female=0
data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])  # S=2, C=0, Q=1 (example)

# Feature selection
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
X = data[features]
y = data['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
