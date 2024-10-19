# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("advertising.csv")

# Display the first few rows of the dataset and basic info
print(df.head())
print(df.describe())
print(f"Dataset shape: {df.shape}")

# Data Visualization
# Pairplot to visualize relationships between features and the target variable 'Sales'
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='scatter')
plt.suptitle("Scatter plots: TV, Radio, Newspaper vs. Sales", y=1.02)
plt.show()

# Histograms for distribution of advertising budgets
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
df['TV'].plot.hist(bins=10, color='blue')
plt.title('TV Advertising Budget')

plt.subplot(1, 3, 2)
df['Radio'].plot.hist(bins=10, color='green')
plt.title('Radio Advertising Budget')

plt.subplot(1, 3, 3)
df['Newspaper'].plot.hist(bins=10, color='purple')
plt.title('Newspaper Advertising Budget')

plt.tight_layout()
plt.show()

# Correlation heatmap to understand relationships between variables
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Splitting the data for training and testing (using 'TV' feature)
X = df[['TV']]
y = df[['Sales']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Display training and testing sets
print("Training data (X_train):\n", X_train.head())
print("\nTraining target (y_train):\n", y_train.head())
print("\nTest data (X_test):\n", X_test.head())
print("\nTest target (y_test):\n", y_test.head())

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Display predicted values
print("Predicted Sales:\n", y_pred)

# Visualizing predictions vs actual data
plt.figure(figsize=(10, 6))

# Scatter plot of actual sales data
plt.scatter(X_test, y_test, color='blue', label='Actual Sales')

# Plot the linear regression line (predictions)
plt.plot(X_test, y_pred, color='red', label='Regression Line')

plt.title('TV Advertising Budget vs. Sales')
plt.xlabel('TV Advertising Budget')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Model coefficients
print(f"Intercept: {model.intercept_[0]}")
print(f"Slope: {model.coef_[0][0]}")
