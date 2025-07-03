import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load Data
df = pd.read_csv('Titanic-Dataset.csv')
print(df.head(10))
print(df.shape)
print(df.describe())
print(df.info())

#Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

#Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

#Bin Age
df['AgeGroup'] = pd.cut(df['Age'], bins=5, labels=False)

#One-Hot Encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

#Drop Irrelevant Columns
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

#Visualizations
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.hist(df['Age'], bins=20)
plt.title('Age Histogram')

plt.subplot(1,3,2)
df.groupby('Sex_male')['Survived'].sum().plot(kind='bar')
plt.title('Survival by Gender')
plt.xticks([0, 1], ['Female', 'Male'], rotation=0)

plt.subplot(1,3,3)
df.boxplot(column='Fare', by='Pclass')
plt.title('Fare by Pclass')
plt.suptitle('') 

plt.tight_layout()
plt.show()

#NumPy Calculations
fare_array = df['Fare'].values
age_array = df['Age'].values

print("Fare - Mean:", np.mean(fare_array), "Median:", np.median(fare_array), "Std:", np.std(fare_array))
print("Age - Mean:", np.mean(age_array), "Median:", np.median(age_array), "Std:", np.std(age_array))

#Min-Max Normalization
df['Fare_norm'] = (fare_array - np.min(fare_array)) / (np.max(fare_array) - np.min(fare_array))
df['Age_norm'] = (age_array - np.min(age_array)) / (np.max(age_array) - np.min(age_array))

#Correlation Matrix
corr_matrix = df.corr(numeric_only=True)
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Correlation Matrix Heatmap')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.tight_layout()
plt.show()

#Save Cleaned Dataset
X = df.drop('Survived', axis=1)
y = df['Survived']
df.to_csv('titanic_cleaned.csv', index=False)
