# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

# Load the dataset
cc_data = pd.read_csv('creditcard.csv')

# Convert non-numeric columns to numeric where applicable
cc_data['Time'] = pd.to_numeric(cc_data['Time'], errors='coerce')
cc_data['Amount'] = pd.to_numeric(cc_data['Amount'], errors='coerce')
cc_data['Class'] = pd.to_numeric(cc_data['Class'], errors='coerce')

# Drop irrelevant non-numeric columns (e.g., merchant info)
non_numeric_columns = ['merchant', 'card_number', 'card_holder', 'address', 'city', 'country', 'zipcode']
cc_data_numeric = cc_data.drop(columns=non_numeric_columns, errors='ignore')

# Drop columns with all missing values
missing_columns = cc_data_numeric.columns[cc_data_numeric.isnull().all()]
cc_data_numeric.drop(columns=missing_columns, inplace=True)

# Impute missing values using the mean strategy
imputer = SimpleImputer(strategy='mean')
cc_data_imputed = pd.DataFrame(imputer.fit_transform(cc_data_numeric), columns=cc_data_numeric.columns)

# Separate features (X) and target (Y)
X = cc_data_imputed.drop(columns='Class')
Y = cc_data_imputed['Class']

# Apply SMOTE for oversampling the minority class (fraud cases)
oversampler = SMOTE()
X_resampled, Y_resampled = oversampler.fit_resample(X, Y)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_resampled, test_size=0.2, 
                                                    stratify=Y_resampled, random_state=2)

# Initialize and train the Logistic Regression model with adjusted max_iter
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Evaluate the model on training data
Y_train_prediction = model.predict(X_train)
training_classification_report = classification_report(Y_train, Y_train_prediction)

# Evaluate the model on test data
Y_test_prediction = model.predict(X_test)
test_classification_report = classification_report(Y_test, Y_test_prediction)

# Combine and print the classification reports
predictions = (
    f"Training Data Classification Report:\n{training_classification_report}\n"
    f"Test Data Classification Report:\n{test_classification_report}"
)

# Output the predictions
print(predictions)
