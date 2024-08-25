import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Replace 'train.csv' and 'test.csv' with the actual file paths
train_data = pd.read_excel('Files\\train.xlsx')
test_data = pd.read_excel('Files\\test.xlsx')

# Features and target variable for training
train_data['TotalBathrooms'] = (train_data['FullBath'] + 0.5 * train_data['HalfBath'] +
                              train_data['BsmtFullBath'] + 0.5 * train_data['BsmtHalfBath'])

train_data['SquareFootage'] = (train_data['GrLivArea'] + 
                              train_data['1stFlrSF'] + train_data['2ndFlrSF'] + 
                              train_data['BsmtFinSF1']+ train_data['BsmtFinSF2']+ train_data['TotalBsmtSF'])

X_train = train_data[['SquareFootage', 'BedroomAbvGr', 'TotalBathrooms']]
y_train = train_data['SalePrice']

# Features for testing (you'll predict the price for these)
test_data['TotalBathrooms'] = (test_data['FullBath'] + 0.5 * test_data['HalfBath'] +
                              test_data['BsmtFullBath'] + 0.5 * test_data['BsmtHalfBath'])

test_data['SquareFootage'] = (test_data['GrLivArea'] + 
                              test_data['1stFlrSF'] + test_data['2ndFlrSF'] + 
                              test_data['BsmtFinSF1']+ test_data['BsmtFinSF2']+ test_data['TotalBsmtSF'])
X_test = test_data[['SquareFootage', 'BedroomAbvGr', 'TotalBathrooms']]

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predicting the prices for the test set
predicted_prices = model.predict(X_test)

# Split the training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train on the split training data
model.fit(X_train_split, y_train_split)

# Predict on the validation set
y_val_pred = model.predict(X_val_split)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_val_split, y_val_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val_split, y_val_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2): {r2}')

# Save the predictions to a CSV file
output = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': predicted_prices
})

# Save the output to a CSV file
output.to_csv('house_price_predictions.csv', index=False)
