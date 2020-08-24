# functions to preprocess data file

# imports
import pandas as pd
from sklearn.model_selection import train_test_split

# sample size
sample_size = 10


# load file
def load_file(file_loc):
    df = pd.read_csv(file_loc)
    return df


# Exploratory data analysis
# get first 5 rows
def get_sample(file_loc):
    df = load_file(file_loc)
    return df.head(10)


# get shape
def get_shape(file_loc):
    df = load_file(file_loc)
    return df.shape


# get statistical description
def get_description(file_loc):
    df = load_file(file_loc)
    return df.describe()


# get total number of nulls in each columns
def get_null_count(file_loc):
    df = load_file(file_loc)
    return df.isnull().sum()


# get null count as percentage of total count
def get_null_count_percentage(file_loc):
    df = load_file(file_loc)
    return df.isnull().sum() / len(df) * 100


# get all columns
def get_all_columns(file_loc):
    df = load_file(file_loc)
    return df.columns


# working with dataframes
# get dataframe object
def get_dataframe(file_loc):
    return load_file(file_loc)


# replace null value by user specified value
def replace_null_value(dataframe, column_name, value):
    dataframe[column_name] = dataframe[column_name].fillna(value)
    return dataframe


# replace all values by user specified values
def replace_value(dataframe, column_name, value):
    dataframe[column_name] = value
    return dataframe


# replace null values by mean
def replace_null_by_mean(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].fillna(dataframe[column_name].mean())
    return dataframe


# change categorical to numerical


# replace null values by median
def replace_null_by_median(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].fillna(dataframe[column_name].median())
    return dataframe


# Subset columns
def get_subset_columns(dataframe, cols):
    return dataframe[cols]


# Drop columns
def drop_columns(dataframe, cols):
    return dataframe.drop(cols,axis=1)


# Split into testing and training
def test_train_split(X, Y, test_size):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=43)
    return x_train, x_test, y_train, y_test


# Get all colums except
def get_all_columns_except(df, column_name):
    df = df.drop(column_name,axis=1)
    return df
