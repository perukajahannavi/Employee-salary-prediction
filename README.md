# Employee-salary-prediction# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data - You can replace this with a CSV file or your dataset
data = {
    'YearsExperience': [1, 2, 3, 4, 5],
    'EducationLevel': [1, 2, 3, 2, 1],  # 1 = High school, 2 = College, 3 = Masters
    'Age': [22, 24, 27, 29, 31],
    'Salary': [40000, 45000, 50000, 55000, 60000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Select the features and target variable
X = df[['YearsExperience', 'EducationLevel', 'Age']]  # Features
y = df['Salary']  # Target

# Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the salary on the test set
y_pred = model.predict(X_test)

# Output the predictions and actual values
print("Predicted Salaries:", y_pred)
print("Actual Salaries:", y_test.values)

# Calculate the Mean Squared Error to evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
 
