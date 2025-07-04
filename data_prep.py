import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

labels=['Survived']
train_data_path = './Titanic_data/train.csv'
test_data_path = './Titanic_data/test.csv'


def load_data(train_data_path, test_data_path):
    """
    Prepares the data for training and testing by loading, cleaning, and normalizing it.
    
    Parameters:
    - train_data_path: Path to the training data CSV file.
    - test_data_path: Path to the testing data CSV file.
    - labels: List of labels/target variables.
    
    Returns:
    - x_data
    - y_data
    - col_names: Column names of the training data.
    """

    with open (train_data_path, 'r') as train_file, open(test_data_path, 'r') as test_file:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)


    print(train_data.head())
    # Display column names
    col_names = train_data.columns.values
    print(col_names)
    return train_data, test_data, col_names



def encode_non_numeric(df, columns):
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

train_data, test_data, col_names = load_data(train_data_path, test_data_path)

print("Choose columns to drop from the training data:")
# Display the columns in the training data
print(train_data.columns)

non_numeric_columns = train_data.select_dtypes(exclude=[np.number]).columns.tolist()
print("Non-numeric columns in training data:", non_numeric_columns)

train_data = encode_non_numeric(train_data, non_numeric_columns)
test_data = encode_non_numeric(test_data, non_numeric_columns)

print("After encoding non-numeric columns:")
print(train_data.head())
print("*"*100)


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



number_of_features = train_data.shape[1] - 1  # Exclude the target variable
print(f"Number of features: {number_of_features}")

number_of_outputs = 1  # User should specify this based on the target variable
print(f"Number of outputs: {number_of_outputs}")

print(test_data.head())

# Convert DataFrame to numpy arrays
x_train = train_data.drop(columns=labels)
y_train = train_data[labels]

x_test = test_data.drop(columns=labels)
y_test = test_data[labels]

# Normalize the data using z-score normalization
x_train_mean = x_train.mean()
x_train_std = x_train.std()
x_train = (x_train - x_train_mean) / x_train_std

x_test_mean = x_test.mean()
x_test_std = x_test.std()
x_test = (x_test - x_test_mean) / x_test_std

print(x_train.head())
# Convert the DataFrame to numpy arrays
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()


# Reshape the data to ensure it has the correct dimensions
train_samples = train_data.shape[0]  # number of samples
test_samples = test_data.shape[0]  # number of samples

# Transpose the input matrix to match the expected shape
x_train = x_train.T
x_test = x_test.T

# Reshape outputs to a matrix
y_train = y_train.reshape(len(labels), train_samples)
y_test = y_test.reshape(len(labels), test_samples)

print(x_train)