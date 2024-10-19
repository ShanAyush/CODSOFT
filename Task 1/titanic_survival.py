# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Loading the Titanic dataset
train = pd.read_csv('Titanic-Dataset.csv')

# Data exploration
# Display the first few rows of the dataset
print(train.head())

# Visualizing missing data using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

# Plotting the distribution of survival count
sns.set_style('whitegrid')
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=train)
plt.title("Survival Count")
plt.show()

# Plotting survival count by gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')
plt.title("Survival Count by Gender")
plt.show()

# Plotting survival count by passenger class
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')
plt.title("Survival Count by Passenger Class")
plt.show()

# Plotting age distribution
plt.figure(figsize=(8, 6))
sns.histplot(train['Age'].dropna(), kde=False, color='darkred', bins=40)
plt.title("Age Distribution")
plt.show()

# Another age distribution plot with different settings
train['Age'].hist(bins=30, color='darkred', alpha=0.3, figsize=(8, 6))
plt.title("Age Distribution Histogram")
plt.show()

# Plotting sibling/spouse count
plt.figure(figsize=(8, 6))
sns.countplot(x='SibSp', data=train)
plt.title("Sibling/Spouse Count")
plt.show()

# Plotting fare distribution
train['Fare'].hist(color='green', bins=40, figsize=(8, 4))
plt.title("Fare Distribution")
plt.show()

# Age imputation based on Pclass using boxplot for visualization
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')
plt.title("Age Distribution by Passenger Class")
plt.show()

# Defining a function to impute missing ages based on passenger class
def impute_age(cols):
    Age, Pclass = cols
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    return Age

# Applying age imputation
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

# Visualizing missing data after imputation
plt.figure(figsize=(10, 6))
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Missing Data After Imputation")
plt.show()

# Dropping the 'Cabin' column due to excessive missing values
train.drop('Cabin', axis=1, inplace=True)

# Dropping rows with remaining missing data
train.dropna(inplace=True)

# Displaying dataset info after cleaning
print(train.info())

# Handling categorical variables: creating dummy variables
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

# Dropping unnecessary columns
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

# Concatenating dummy variables with the cleaned dataset
train = pd.concat([train, sex, embark], axis=1)

# Checking the cleaned dataset
print(train.head())

# Preparing data for model training
X = train.drop('Survived', axis=1)
y = train['Survived']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Logistic regression model training
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# Making predictions
predictions = logmodel.predict(X_test)

# Model evaluation
# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", conf_matrix)

# Accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy Score: {accuracy * 100:.2f}%")

# Classification report
print("Classification Report:\n", classification_report(y_test, predictions))
