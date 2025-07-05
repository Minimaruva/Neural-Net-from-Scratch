import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

labels=['Survived']
train_data_path = './Titanic_data/train.csv'
test_data_path = './Titanic_data/test.csv'


def load_data_train_test(train_data_path, test_data_path):
    with open (train_data_path, 'r') as train_file, open(test_data_path, 'r') as test_file:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)


    print(train_data.head())
    # Display column names
    col_names = train_data.columns.values
    print(col_names)
    return train_data, test_data, col_names



def encode_non_numeric(df):
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in non_numeric_columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def fill_missing_values(df):
    return df.fillna(df.median())

def normalize_data_z_score(df, labels=None):
    if labels is not None:
        features = df.drop(columns=labels)
        labels_df = df[labels] 

        features_norm = (features - features.mean()) / features.std()
        df = pd.concat([features_norm, labels_df], axis=1)
    else:
        df = (df - df.mean()) / df.std()
    
    return df


def split_reshape_data(df, labels):
    samples = df.shape[0]  # number of samples

    x_data = df.drop(columns=labels).to_numpy()
    y_data = df[labels].to_numpy()

    x_data = x_data.T 
    y_data = y_data.reshape(len(labels), samples)

    return x_data, y_data

train_data, test_data, col_names = load_data_train_test(train_data_path, test_data_path)

print("Choose columns to drop from the training data:")
# Display the columns in the training data
print(train_data.columns)


print("Non-numeric columns in training data:", train_data.select_dtypes(exclude=[np.number]).columns.tolist())

train_data = encode_non_numeric(train_data)
test_data = encode_non_numeric(test_data)

print("After encoding non-numeric columns:")
print(train_data.head())
print("*"*100)


# Immediately drop non-numeric columns
## TODO: show user non-numeric columns and ask if they want to drop them or encode them


train_data = train_data.select_dtypes(include=[np.number])
test_data = test_data.select_dtypes(include=[np.number])

print(train_data.head())

# Fill missing values with the mean of each column
## ToDo tell user about missing values and let them choose how to handle them
print("Missing values in train data:\n", train_data.isnull().sum())
print("Missing values in test data:\n", test_data.isnull().sum())

train_data = fill_missing_values(train_data)
test_data = fill_missing_values(test_data)


# Normalise the data
## ToDo tell user about normalisation and let them choose the method
### https://developers.google.com/machine-learning/crash-course/numerical-data/normalization



number_of_features = train_data.shape[1] - 1  # Exclude the target variable
print(f"Number of features: {number_of_features}")

number_of_outputs = 1  # User should specify this based on the target variable
print(f"Number of outputs: {number_of_outputs}")

print(test_data.head())

train_data = normalize_data_z_score(train_data, labels)
test_data = normalize_data_z_score(test_data, labels)

x_train, y_train = split_reshape_data(train_data, labels)
x_test, y_test = split_reshape_data(test_data, labels)

# # Convert DataFrame to numpy arrays
# x_train = train_data.drop(columns=labels)
# y_train = train_data[labels]

# x_test = test_data.drop(columns=labels)
# y_test = test_data[labels]


# # Convert the DataFrame to numpy arrays
# x_train = x_train.to_numpy()
# y_train = y_train.to_numpy()
# x_test = x_test.to_numpy()
# y_test = y_test.to_numpy()


# # Reshape the data to ensure it has the correct dimensions
# train_samples = train_data.shape[0]  # number of samples
# test_samples = test_data.shape[0]  # number of samples

# # Transpose the input matrix to match the expected shape
# x_train = x_train.T
# x_test = x_test.T

# # Reshape outputs to a matrix
# y_train = y_train.reshape(len(labels), train_samples)
# y_test = y_test.reshape(len(labels), test_samples)

print(x_train)