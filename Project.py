# Import only the required libraries
import pandas as pd  # For data handling
import numpy as np  # For numerical operations
import seaborn as sns  # For visualization
import matplotlib.pyplot as plt  # For plotting
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LinearRegression  # For regression model
from sklearn.metrics import mean_absolute_error, r2_score  # For evaluation metrics

# Load/Attach the dataset
data = pd.read_csv('Salary Data.csv')

# Preview the dataset
print(data)

#To check the unique values of the Education Level we have TWO options
data['Education Level'].value_counts()
#OR
data['Education Level'].unique() #here we also get to see the nan{Is there any (Null) Values}

# Drop columns that won't be used for prediction
data = data.drop(['Job Title'], axis=1)
data = data.drop(['Education Level'], axis=1)

# Check for missing values
print(data.isnull().sum())

# Fill or drop missing values. We will drop the missing value to target the accuracy.
data = data.dropna()  # Drop rows with missing values

data.info()

data.describe()

data

# Convert 'Gender' and 'Education Level' into numerical values
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})  # Gender encoding


# Calculate correlations
corr = data.corr()


# Set random seed for reproducibility
# Define features (X) and target (y)
np.random.seed(123)
X = data[['Years of Experience']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

# Scatter plot for predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolors='w', s=100, label='Predicted vs Actual')

# Add a line of equality
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Line of Equality')

# Labels and title
plt.xlabel('Actual Salary', fontsize=14)
plt.ylabel('Predicted Salary', fontsize=14)
plt.title('Actual vs Predicted Salary', fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()