"""
pip install "git+https://github.com/nixtla/neuralforecast.git@main"
pip install darts
"""

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae, smape
from darts.utils.timeseries_generation import datetime_attribute_timeseries, holidays_timeseries
from sklearn.preprocessing import LabelEncoder

# import dataframe
Y_df = pd.read_csv('Y_df.csv')

# add timeseries covariates
Y_df['datetime'] = pd.to_datetime(Y_df['ds'])
Y_df.drop('ds', axis=1, inplace=True)
Y_df.set_index('datetime', inplace=True)
series = TimeSeries.from_series(Y_df)

series = series.add_datetime_attribute('hour')
series = series.add_datetime_attribute('dayofweek')
series = series.add_datetime_attribute('month')
series = series.add_datetime_attribute('quarter')
series = series.add_datetime_attribute('day')
series = series.add_datetime_attribute('year')
series = series.add_holidays(country_code='ITA')

Y_df = TimeSeries.pd_dataframe(series).reset_index()
Y_df.rename({'datetime': 'ds'}, axis=1, inplace=True)

df_4 = Y_df[['y', 'hour', 'dayofweek', 'month', 'holidays', 'day', 'year', 'psvda', 'load forecast']]

# Create past regressor columns
num_past_periods = 168  # Number of past observed periods

# past gas
df_4[f'psvda_t-{24}'] = Y_df['psvda'].shift(24)

# target
for i in range(1, 24):
    df_4[f'y_t+{i}'] = Y_df['y'].shift(-i)

# future load
for i in range(0, 24):
    df_4[f'load_forecast_t+{i}'] = Y_df['load forecast'].shift(-i)

for i in range(1, 169):
    df_4[f'y_t-{i}'] = Y_df['y'].shift(i)

df_4.rename({'y': 'y_t+0'}, axis=1, inplace=True)
# Remove rows with NaN values due to shifting
df_4 = df_4.dropna()
df_4['weekday-hour'] = df_4['dayofweek']*df_4['hour']
df_4['month-hour'] = df_4['month']*df_4['hour']
df_4['year'] = [x[:4] for x in df_4.index.astype('str')]
df_4[['hour', 'dayofweek', 'month', 'holidays', 'day', 'year', 'weekday-hour', 'month-hour']] = df_4[['hour', 'dayofweek', 'month', 'holidays', 'day', 'year', 'weekday-hour', 'month-hour']].astype(np.int32)


df_4.reset_index(inplace=True)
def is_bridge_day(day, holiday, dayofweek):
    if holiday == 1:
        return 7
    elif dayofweek in [5,6] and holiday == 0:
        return dayofweek
    elif dayofweek == 0 and holiday == 0 and df_4.iloc[day+24]['holidays'] == 1:
        return 8
    elif dayofweek == 4 and holiday == 0 and df_4.iloc[day-24]['holidays'] == 1:
        return 8
    else:
        return dayofweek

df_4['day_of_week'] = np.vectorize(is_bridge_day)(df_4.index, df_4['holidays'], df_4['dayofweek'])


exog_past = [f'y_t-{i}' for i in range(1, 169)]
exog_past.append('psvda_t-24')

exog_fut = [f'load_forecast_t+{i}' for i in range(0, 24)]
exog = exog_past + exog_fut

target = [f'y_t+{i}' for i in range(24)]

calendar = ['hour', 'day_of_week', 'month', 'day', 'year']
df_4.set_index('ds', inplace=True)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

encoder = LabelEncoder()

for cal in calendar:
  df_4[cal] = encoder.fit_transform(df_4[cal]).astype(np.int32)

# Determine the indices for splitting the dataset
test_index = len(df_4) - 190*24
validation_index = test_index - 190*24

# Split the dataset into training, validation, and test sets
Y_train_df = df_4.iloc[:validation_index]
Y_val_df = df_4.iloc[validation_index:test_index]
Y_test_df = df_4.iloc[test_index:]

from sklearn.preprocessing import MinMaxScaler
scaler_1 = MinMaxScaler()
scaler_2 = MinMaxScaler()
variables = exog + calendar

scaler_1.fit(Y_train_df[exog])
scaler_2.fit(Y_train_df[target])

# training set scaled
Y_train_scaled_exog = pd.DataFrame(scaler_1.transform(Y_train_df[exog]), columns=Y_train_df[exog].columns, index=Y_train_df.index)
Y_train_scaled_df = pd.concat([Y_train_scaled_exog, Y_train_df[calendar]], axis=1)
Y_train_scaled_y = pd.DataFrame(scaler_2.transform(Y_train_df[target]), columns=Y_train_df[target].columns)
# test set scaled
Y_test_scaled_exog = pd.DataFrame(scaler_1.transform(Y_test_df[exog]), columns=Y_test_df[exog].columns, index=Y_test_df.index)
Y_test_scaled_df = pd.concat([Y_test_scaled_exog, Y_test_df[calendar]], axis=1)
Y_test_scaled_y = pd.DataFrame(scaler_2.transform(Y_test_df[target]), columns=Y_test_df[target].columns)

# validation set scaled
Y_val_scaled_exog = pd.DataFrame(scaler_1.transform(Y_val_df[exog]), columns=Y_val_df[exog].columns, index=Y_val_df.index)
Y_val_scaled_df = pd.concat([Y_val_scaled_exog, Y_val_df[calendar]], axis=1)
Y_val_scaled_y = pd.DataFrame(scaler_2.transform(Y_val_df[target]), columns=Y_val_df[target].columns)

# train + validation set scaled
Y_trainval_scaled_df = pd.concat([Y_train_scaled_df, Y_val_scaled_df], axis=0)
Y_trainval_scaled_y = pd.concat([Y_train_scaled_y, Y_val_scaled_y], axis=0)


from sklearn.preprocessing import LabelEncoder
# Sample dat
# Assume you have a DataFrame called 'df' containing the relevant data including calendar variables and percentage deviations.

# Define input shapes
num_calendar_features = len(calendar)  # Number of calendar features: hour, day_of_month, day_of_week, holiday, month, quarter
num_exog_features = len(exog)  # Number of past observed percentage deviations to consider as input

# Prepare train input data
calendar_features = Y_train_scaled_df[calendar].values

past_regressors = Y_train_scaled_df[exog].values

target_variable = Y_train_scaled_y[target].values

# Prepare val input data
calendar_features_val = Y_val_scaled_df[calendar].values

past_regressors_val = Y_val_scaled_df[exog].values

target_variable_val = Y_val_scaled_y[target].values

#Prepare test input data
test_calendar_features = Y_test_scaled_df[calendar].values
test_past_regressors = Y_test_scaled_df[exog].values

# training + validation input data
trainval_calendar_features = Y_trainval_scaled_df[calendar].values

trainval_past_regressors = Y_trainval_scaled_df[exog].values

trainval_target = Y_trainval_scaled_y[target].values


