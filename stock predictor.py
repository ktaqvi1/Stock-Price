#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Input the company name
company = input("Enter a string: ")
path = "C:\\Users\\bab\\Documents\\stockPredictionProject\\" + company + ".csv"
print(path)

# Load the dataset
df = pd.read_csv(path)
df.head()

# Convert 'Date' to datetime format and drop irrelevant columns
df['Date'] = pd.to_datetime(df['Date'])
df.drop('Adj Close', axis=1, inplace=True)

# Ensure correct data types
df['Volume'] = df['Volume'].astype(float)

# Handle missing values by dropping rows with NaN values
df_new = df[df.select_dtypes(include=[np.number]).apply(np.isfinite).all(1)]


# Plot the 'Open' price over time
df_new['Open'].plot(figsize=(16, 6))

# Define features and target variable
x = df_new[['Open', 'High', 'Low', 'Volume']]
y = df_new['Close']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Initialize and train the Linear Regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Print the model coefficients and intercept
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)

# Make predictions on the test set
predicted = regressor.predict(x_test)

# Create a DataFrame to compare actual vs predicted values
dfr = pd.DataFrame({'Actual': y_test, 'Predicted': predicted})

# Display actual vs predicted values
print(dfr)

# Sort the DataFrame by actual and predicted values for better comparison
dfr_sorted_actual = dfr.sort_values("Actual", ascending=False)
dfr_sorted_predicted = dfr.sort_values("Predicted", ascending=False)

# Visualize the actual closing prices
lst = [i for i in range(0, len(dfr_sorted_actual))]
plt.figure(figsize=(12, 6))
plt.plot(lst, dfr_sorted_actual['Actual'], label='Actual Closing Price', color='blue')
plt.xlabel('Time')
plt.ylabel('Stock Closing Price')
plt.title('Actual Closing Price')
plt.grid(True)
plt.show()

# Visualize the predicted closing prices
plt.figure(figsize=(12, 6))
plt.plot(lst, dfr_sorted_predicted['Predicted'], label='Predicted Closing Price', color='yellow')
plt.xlabel('Time')
plt.ylabel('Stock Closing Price')
plt.title('Predicted Closing Price')
plt.grid(True)
plt.show()

# Combined plot: Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(lst, dfr_sorted_actual['Actual'], label='Actual Closing Price', color='blue')
plt.plot(lst, dfr_sorted_predicted['Predicted'], label='Predicted Closing Price', color='yellow')
plt.xlabel('Time')
plt.ylabel('Stock Closing Price')
plt.title('Actual vs Predicted Closing Prices')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model using R-squared, Mean Absolute Error, Mean Squared Error, and RMSE
r2_score = regressor.score(x_test, y_test)
mae = mean_absolute_error(y_test, predicted)
mse = mean_squared_error(y_test, predicted)
rmse = np.sqrt(mse)

print("R-squared:", r2_score)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)


# In[ ]:




