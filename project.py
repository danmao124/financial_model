#!/usr/bin/env python
# coding: utf-8

# In[98]:


#from google.colab import drive
# drive.mount('/content/drive')


# In[99]:


import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import warnings
import pickle
import pandas as pd
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError
from datetime import timedelta
import datetime
from sklearn.model_selection import GridSearchCV

# Please read, my idea was to classify the data into three cateogries: "UP", "DOWN", "NEUTRAL"
# UP - the price of the stock is up MINIMUM_GAIN percent after LOOK_AHEAD_DAYS days.
# DOWN - the price of the stock is down MINIMUM_GAIN percent after LOOK_AHEAD_DAYS days.
# NEUTRAL - the price of stock is did not rise or fall past the MINIMUM_GAIN threshhold
#
# EXAMPLE: LOOK_AHEAD_DAYS = 30, MINIMUM_GAIN = .05 (5%). If stock is up over 5% ore more after 30 days,
# we mark it as 'UP'. If the stock is down 5% or more after 30 days, we mark it as 'DOWN'. If the stock
# is neither up nor down 5%, then we mark it as 'NEUTRAL'

# MINIMUM_GAIN = .05 #minimal gain to be considered up or down for classification, UNUSED NOW

LOOK_BACK_DAYS = 30  # number of days into the past we would like to take into account
LOOK_AHEAD_DAYS = 1  # number of days into the future we are trying to predict


class Utilities:

    @staticmethod
    def getData(ticker, start_date, end_date):
        try:
            stock_data = data.DataReader(ticker,
                                         'yahoo',
                                         start_date,
                                         end_date)
            return stock_data
        except RemoteDataError:
            print('No data found for {t}'.format(t=ticker))


# just get the btc price points from the past year
start_date = datetime.datetime.now() - timedelta(365)
end_date = datetime.datetime.now() - timedelta(1)
btc_df = Utilities.getData(
    'BTC-USD', str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d')))
btc_df


# In[100]:


def createLookBackCols():
    for i in range(LOOK_BACK_DAYS):
        num_rows = btc_df.shape[0]
        num_cols = btc_df.shape[1]
        btc_df.insert(loc=num_cols, column='High_Past_' +
                      str(i + 1), value=['N/A'] * num_rows)
        btc_df.insert(loc=num_cols, column='Low_Past_' +
                      str(i + 1), value=['N/A'] * num_rows)
        btc_df.insert(loc=num_cols, column='Open_Past_' +
                      str(i + 1), value=['N/A'] * num_rows)
        btc_df.insert(loc=num_cols, column='Close_Past_' +
                      str(i + 1), value=['N/A'] * num_rows)
        btc_df.insert(loc=num_cols, column='Volume_Past_' +
                      str(i + 1), value=['N/A'] * num_rows)
        btc_df.insert(loc=num_cols, column='Adj Close_Past_' +
                      str(i + 1), value=['N/A'] * num_rows)


createLookBackCols()  # create additional look back columns
btc_df = btc_df.loc[~btc_df.index.duplicated(
    keep='first')]  # delete all duplicate indices

for index, row in btc_df.iterrows():
    current_date = index.to_pydatetime()
    if (current_date - start_date).days >= LOOK_BACK_DAYS - 1:
        for i in range(LOOK_BACK_DAYS):
            index = i + 1
            look_back_date = current_date - timedelta(days=index)

            while True:  # we need to keep subtracting days to the look_back_date because there is no index for that day
                if look_back_date in btc_df.index:
                    break
                look_back_date = look_back_date - timedelta(days=1)

            btc_df.at[pd.Timestamp(current_date), 'High_Past_' + str(index)
                      ] = btc_df.at[pd.Timestamp(look_back_date), 'High']
            btc_df.at[pd.Timestamp(current_date), 'Low_Past_' + str(index)
                      ] = btc_df.at[pd.Timestamp(look_back_date), 'Low']
            btc_df.at[pd.Timestamp(current_date), 'Open_Past_' + str(index)
                      ] = btc_df.at[pd.Timestamp(look_back_date), 'Open']
            btc_df.at[pd.Timestamp(current_date), 'Close_Past_' + str(index)
                      ] = btc_df.at[pd.Timestamp(look_back_date), 'Close']
            btc_df.at[pd.Timestamp(current_date), 'Volume_Past_' + str(index)
                      ] = btc_df.at[pd.Timestamp(look_back_date), 'Volume']
            btc_df.at[pd.Timestamp(current_date), 'Adj Close_Past_' + str(
                index)] = btc_df.at[pd.Timestamp(look_back_date), 'Adj Close']

btc_df = btc_df[btc_df.High_Past_1 != 'N/A']
btc_df


# In[101]:


# introduce new column that will be the trend we are predicting
btc_df = btc_df.assign(MonthTrend=lambda x: "N/A")
btc_df = btc_df.loc[~btc_df.index.duplicated(
    keep='first')]  # delete all duplicate indices

# Populate the classification column MonthTrend
for index, row in btc_df.iterrows():
    current_date = index.to_pydatetime()

    if (end_date - current_date).days > LOOK_AHEAD_DAYS:
        look_ahead_date = current_date + timedelta(days=LOOK_AHEAD_DAYS)

        while True:  # we need to keep adding days to the look_ahead_date because there is no index for that day
            if look_ahead_date in btc_df.index:
                break
            look_ahead_date = look_ahead_date + timedelta(days=1)

        # Here we check if the open price in the future is higher than today's open price.
        if btc_df.loc[pd.Timestamp(look_ahead_date)]['Open'] > btc_df.loc[index]['Open']:
            btc_df.at[pd.Timestamp(look_ahead_date), 'MonthTrend'] = 'UP'
        else:
            btc_df.at[pd.Timestamp(look_ahead_date), 'MonthTrend'] = 'DOWN'

# Now delete all columns that have a MonthTrend value of N/A
btc_df = btc_df[btc_df.MonthTrend != 'N/A']

# Data is now fully processed and ready to be trained on the machine learning model
# There are 3 values for MonthTrend: UP, DOWN,NEUTRAL
btc_df


# In[102]:


def generateFeatures(features):
    complete_features = []
    for feature in features:
        complete_features.append(feature)
        for i in range(LOOK_BACK_DAYS):
            complete_features.append(feature + "_Past_" + str(i + 1))
    return complete_features


features = generateFeatures(['Close', 'Open'])
features.append('MonthTrend')

# selecting the important feature open close
btc = btc_df[features]
btc


# In[103]:


# load pre-fetched news sentiment data and add in the dataframe
warnings.filterwarnings('ignore')
f = open(os.path.join(os.getcwd(), "polaritys_score.pkl"), "rb")
#f = open("./drive/My Drive/final_project/polaritys_score.pkl", "rb")
polaritys_file = pickle.load(f)
f.close()

polaritys_df = pd.DataFrame(index=list(polaritys_file.keys())[
                            ::-1], columns=['neg', 'neu', 'pos'])
for date in list(polaritys_file.keys())[::-1]:
    for polar in ['neg', 'neu', 'pos']:
        polaritys_df.loc[date][polar] = polaritys_file[date][polar]

# keep data only after 2019-09-09 and remove data on 2020-08-03(error in Yahoo?)
btc_index_str = [btc_index.strftime('%Y-%m-%d')
                 for btc_index in list(btc.index)]

sentiment_data = polaritys_df.loc[btc_index_str]
sentiment_data.columns = ['negative', 'neutral', 'positive']

# add sentiment in the dataframe
btc[['negative', 'neutral', 'positive']
    ] = sentiment_data[['negative', 'neutral', 'positive']]
btc

le = LabelEncoder()

i = 5
train_df = btc[i: i + LOOK_BACK_DAYS + 1]
X_TRAIN = train_df.drop(['MonthTrend'], axis=1)
Y_TRAIN = le.fit_transform(train_df['MonthTrend'])
test_df = btc[i + LOOK_BACK_DAYS + LOOK_AHEAD_DAYS: i +
              LOOK_BACK_DAYS + LOOK_AHEAD_DAYS + 1]
X_TEST = test_df.drop(['MonthTrend'], axis=1)
results_df = btc[i + LOOK_BACK_DAYS + 2 *
                 LOOK_AHEAD_DAYS: i + LOOK_BACK_DAYS + 2*LOOK_AHEAD_DAYS + 1]

model_regression = LogisticRegression()
param_grid = {
    'C': [1, 2, 4, 6, 8, 10],
    'penalty': ["l1", "l2"],
    'fit_intercept': [True, False],
    'class_weight': ["balanced", None],
    'warm_start': [True, False]}   # simplified the CV to reduce the strain on my PC
regression_grid = GridSearchCV(model_regression, param_grid, cv=TimeSeriesSplit(
    max_train_size=None, n_splits=10), verbose=1, n_jobs=6)
regression_grid.fit(X_TRAIN, Y_TRAIN)
model_regresssion = regression_grid.best_estimator_

prediction = model_regresssion.predict(X_TEST)[0]
if prediction == 1:
    print(" Bitcoin price is predicted to go up tomorrow")
else:
    print(" Bitcoin price is predicted to go down tomorrow")


# %%
