import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data_train_test(train_data_path, test_data_path):
    with open (train_data_path, 'r') as train_file, open(test_data_path, 'r') as test_file:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)


    col_names = train_data.columns.values
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

        mean = features.mean()
        std = features.std()

        features_norm = (features - mean) / std
        df = pd.concat([features_norm, labels_df], axis=1)
    else:
        mean = df.mean()
        std = df.std()
        df = (df - mean) / std
    
    return df, mean, std


def split_reshape_data(df, labels):
    samples = df.shape[0]  # number of samples

    x_data = df.drop(columns=labels).to_numpy()
    y_data = df[labels].to_numpy()

    x_data = x_data.T 
    y_data = y_data.reshape(len(labels), samples)

    return x_data, y_data


# Display the columns in the training data


# Immediately drop non-numeric columns
## TODO: show user non-numeric columns and ask if they want to drop them or encode them

# Fill missing values with the mean of each column
## ToDo tell user about missing values and let them choose how to handle them


# Normalise the data
## ToDo tell user about normalisation and let them choose the method
### https://developers.google.com/machine-learning/crash-course/numerical-data/normalization


