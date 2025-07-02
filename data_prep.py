import numpy as np
import pandas as pd



train_data_path = './Titanic_data/train.csv'
test_data_path = './Titanic_data/test.csv'

with open (train_data_path, 'r') as train_file, open(test_data_path, 'r') as test_file:
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)


print(train_data.head())
# Display column names
col_names = train_data.columns.values
print(col_names)

# Immediately drop non-numeric columns
## TODO: show user non-numeric columns and ask if they want to drop them or encode them
'''
Potential solution for encoding non-numeric columns:
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(df['COL1'])
df['COL1'] = enc.transform(df['col1'])
'''

train_data = train_data.select_dtypes(include=[np.number])
test_data = test_data.select_dtypes(include=[np.number])

print(train_data.head())

# Fill missing values with the mean of each column
## ToDo tell user about missing values and let them choose how to handle them
print("Missing values in train data:\n", train_data.isnull().sum())
print("Missing values in test data:\n", test_data.isnull().sum())

train_data = train_data.fillna(train_data.mean())
test_data = test_data.fillna(test_data.mean())

# Normalise the data
## ToDo tell user about normalisation and let them choose the method
### https://developers.google.com/machine-learning/crash-course/numerical-data/normalization

# Normalize the data using z-score normalization
train_mean = train_data.mean()
train_std = train_data.std()
train_data = (train_data - train_mean) / train_std

test_mean = test_data.mean()
test_std = test_data.std()
test_data = (test_data - test_mean) / test_std

number_of_features = train_data.shape[1] - 1  # Exclude the target variable
print(f"Number of features: {number_of_features}")

number_of_outputs = 1  # User should specify this based on the target variable
print(f"Number of outputs: {number_of_outputs}")

# Convert DataFrame to numpy arrays