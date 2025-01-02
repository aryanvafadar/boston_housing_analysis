from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


"""
CSV File Import 
"""
current_directory = Path(__file__).parent
housing_csv = current_directory.joinpath('BostonHousing.csv')


"""
Create & Quickly Clean Frame
"""
frame = pd.read_csv(filepath_or_buffer=housing_csv)
frame.columns = frame.columns.str.strip() # remove all whitespace in column names

# Display some basic info
print(f"Column Names: {np.array(frame.columns)}")
print(f"Number of Features: {len(frame.columns) - 1}")
print(f"Label Name: {frame.columns[-1]} ")
print(f"Frame Shape: {frame.shape}")
print(f"Frame Size: {frame.size}")


"""
Create the Features and Labels
"""
# X = Is our features, which represent the independent variables
scaler = StandardScaler()
X = frame[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
X = scaler.fit_transform(X=X)
print(X)

# y = Is our label, which represents the dependent variable and what we will try to predict
y = frame[['medv']]
print(y)

print(X.shape)
print(y.shape)