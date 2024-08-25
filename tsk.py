import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
train_data = pd.read_excel('Files\\train.xlsx')
test_data = pd.read_excel('Files\\test.xlsx')

# Feature engineering
train_data['TotalBathrooms'] = (train_data['FullBath'] + 0.5 * train_data['HalfBath'] +
                                train_data['BsmtFullBath'] + 0.5 * train_data['BsmtHalfBath'])
train_data['SquareFootage'] = (train_data['GrLivArea'] + 
                                train_data['1stFlrSF'] + train_data['2ndFlrSF'] + 
                                train_data['BsmtFinSF1'] + train_data['BsmtFinSF2'] + train_data['TotalBsmtSF'])

test_data['TotalBathrooms'] = (test_data['FullBath'] + 0.5 * test_data['HalfBath'] +
                                test_data['BsmtFullBath'] + 0.5 * test_data['BsmtHalfBath'])
test_data['SquareFootage'] = (test_data['GrLivArea'] + 
                                test_data['1stFlrSF'] + test_data['2ndFlrSF'] + 
                                test_data['BsmtFinSF1'] + test_data['BsmtFinSF2'] + test_data['TotalBsmtSF'])

X_train = train_data[['SquareFootage', 'BedroomAbvGr', 'TotalBathrooms']]
y_train = train_data['SalePrice']

X_test = test_data[['SquareFootage', 'BedroomAbvGr', 'TotalBathrooms']]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train models
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Cross-validation for Linear Regression
cv_scores = cross_val_score(linear_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores.mean())
print(f'Cross-Validated RMSE for Linear Regression: {cv_rmse}')

# Train and evaluate Ridge Regression
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = ridge_model.predict(X_test_scaled)
print(f'Ridge Model R-squared: {r2_score(y_train, ridge_model.predict(X_train_scaled))}')

# Train and evaluate Lasso Regression
lasso_model.fit(X_train_scaled, y_train)
lasso_pred = lasso_model.predict(X_test_scaled)
print(f'Lasso Model R-squared: {r2_score(y_train, lasso_model.predict(X_train_scaled))}')

# Train and evaluate Random Forest
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(f'Random Forest R-squared: {r2_score(y_train, rf_model.predict(X_train))}')

# Save predictions
output = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': rf_pred  # Use the best model based on validation performance
})

output.to_csv('house_price_predictions.csv', index=False)
