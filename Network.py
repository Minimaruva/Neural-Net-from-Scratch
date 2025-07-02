import numpy as np
import pandas as pd

with open ('./Titanic_data/train.csv', 'r') as file:
    data = pd.read_csv(file)


# Display column names
col_names = data.columns.values
print(col_names)

# Immediately drop non-numeric columns