import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
sample = pd.read_csv("C:\\backup\\Prodigy_Infotech\\house-prices-advanced-regression-techniques\\sample_submission.csv")
print(sample.head())
train = pd.read_csv("C:\\backup\\Prodigy_Infotech\\house-prices-advanced-regression-techniques\\test.csv")
test = pd.read_csv("C:\\backup\\Prodigy_Infotech\\house-prices-advanced-regression-techniques\\train.csv")
df = pd.concat([train,test],ignore_index = True)
price = df.pop('SalePrice')
print(df.head())
print(f'training data shape {train.shape} and the testing data shape {test.shape} and the shape of dataset is {df.shape}')
print(df.info())
pd.set_option('display.max_rows', 10)
print(df.isna().sum())
df.drop(columns = ['Alley','PoolQC','Fence','MiscFeature'],inplace = True)
print(df.describe())
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
print(df['BsmtQual'].unique())
df['BsmtQual'] = df['BsmtQual'].fillna('NoBsmt')
print(df['BsmtQual'].unique())
df['BsmtCond'] = df['BsmtCond'].fillna('NoBsmt')
df['BsmtExposure'] = df['BsmtExposure'].fillna('NoBsmt')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('NoBsmt')
df['BsmtFinType2'] = df['BsmtFinType2'].fillna('NoBsmt')
print(df['Electrical'].value_counts())
df['Electrical'] = df['Electrical'].fillna('Unknown')
print(df['FireplaceQu'].value_counts())
df['FireplaceQu'] = df['FireplaceQu'].fillna('NoFireplace')
print(df['GarageType'].value_counts())
df['GarageType'] = df['GarageType'].fillna('NoGarage')
print(df['GarageYrBlt'].unique())
print(df['YearBuilt'].unique())
correlation = df['YearBuilt'].corr(df['GarageYrBlt'])
print("Correlation: ",correlation)
df.loc[df['GarageYrBlt'].isna(), 'GarageYrBlt'] = df['YearBuilt']
df['GarageFinish'] = df['GarageFinish'].fillna('NoGarage')
df['GarageQual'] = df['GarageQual'].fillna('NoGarage')
df['GarageCond'] = df['GarageCond'].fillna('NoGarage')
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# Identify missing values
missing_values = df.isna().sum()
columns_with_missing = missing_values[missing_values > 0]
print(columns_with_missing)

# Determine data types of columns with missing values
data_types = df.dtypes[columns_with_missing.index]
print(data_types)

# Iterate through columns with missing values
missing_columns = columns_with_missing.index
for column in missing_columns:
    if data_types[column] == 'object':  # Categorical column
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)
    else:  # Numerical column
        mean_value = df[column].mean()
        df[column] = df[column].fillna(mean_value)
print(len(df[df.duplicated()]))

df['SalePrice'] = price
# Plot histograms
df.hist(figsize=(20, 15), bins=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(22, 8))
sns.boxplot(x='Neighborhood', y='SalePrice', data=df)
plt.xticks(rotation=45)
plt.title('Box Plot of House Prices by Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Sale Price')
plt.grid(True)
plt.show()


numerical_cols = df.select_dtypes(include=['number']).columns
numerical_df = df[numerical_cols]

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

# Plot the heatmap
plt.figure(figsize=(22, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
plt.title('Scatter Plot of Sale Price vs. Living Area')
plt.xlabel('GrLivArea (Square Feet)')
plt.ylabel('Sale Price')
plt.grid(True)
plt.show()

# Function to display unique values of each categorical column
def display_unique_values(df):
    for column in df.select_dtypes(include=['object']).columns:
        unique_values = df[column].unique()
        print(f"Unique values in '{column}': {unique_values}")

# Call the function
print(display_unique_values(df))

print(df.columns)

# One-Hot Encoding for nominal categories

one_hot_cols = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
    'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
    'Foundation', 'Heating', 'CentralAir', 'Electrical', 'Functional',
    'GarageType', 'GarageFinish', 'PavedDrive', 'SaleType', 'SaleCondition'
]
df = pd.get_dummies(df, columns=one_hot_cols)


from sklearn.preprocessing import LabelEncoder, StandardScaler

# Label Encoding for ordinal categories
label_cols = [
    'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
    'GarageQual', 'GarageCond'
]

label_encoders = {}  # Dictionary to store the label encoders
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store the encoder for the column

numerical_cols = df.select_dtypes(include=['number']).columns
numerical_cols = numerical_cols.drop('Id')
# Create a copy of the DataFrame
df_standardized = df.copy()

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
# Display the resulting DataFrame
print(df)

# Create training and testing DataFrames
train_df = df[~df['SalePrice'].isna()].copy()
test_df = df[df['SalePrice'].isna()].copy()

# Drop 'SalePrice' column from test_df
test_df.drop(columns=['SalePrice'], inplace=True)
X = train_df.drop(columns = ['SalePrice'])
y = train_df['SalePrice']
print(test_df.shape)
print(X.shape)



from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
print(f'Mean Squared Error: {mse}')


# Make predictions
test_predictions = model.predict(test_df)

# Create a submission file
submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_predictions})
submission.shape


submission.to_csv('submission.csv', index=False)

















