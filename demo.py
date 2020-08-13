
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import json
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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

# fetch news of today's and create wordcloud of the keywords
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')


def today_news():
    toPublishedDate = datetime.datetime.today().strftime('%Y-%m-%d')
    fromPublishedDate = (datetime.datetime.today() -
                         datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/Search/NewsSearchAPI"
    querystring = {"autoCorrect": "false", "pageNumber": "1", "pageSize": "30", "q": "Bitcoin",
                   "safeSearch": "false", "fromPublishedDate": fromPublishedDate, "toPublishedDate": toPublishedDate}
    headers = {
        'x-rapidapi-host': "contextualwebsearch-websearch-v1.p.rapidapi.com",
        'x-rapidapi-key': "6e64d32139msh66d110fec2ee5d6p1b2f8ajsn506563d39c4e"
    }
    response = requests.request(
        "GET", url, headers=headers, params=querystring)
    response_json = response.json()['value']

    sid = SentimentIntensityAnalyzer()
    general_polarity = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
    scaled_polarity = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0}

    descriptions = []
    for i in range(len(response_json)):
        description = response_json[i]['description']
        if description:
            non_symbols = re.sub(r'[^\w]', ' ', description)  # remove symbols
            non_digits = re.sub(r'\d+', ' ', non_symbols)  # remove digits
            # remove extra whitespaces
            one_space = re.sub(' +', ' ', non_digits)
            description_lower = one_space.lower()  # transform characters into lower-case
            description_words = ' '.join(
                w for w in description_lower.split() if len(w) > 1)  # remove single characters
            descriptions.append(description_words)  # save descriptions context
            polarity = sid.polarity_scores(
                description_words)  # obtain polarities
            for polar in ['neg', 'neu', 'pos']:
                general_polarity[polar] += polarity[polar]

    for polar in general_polarity:
        scaled_polarity[polar] = general_polarity[polar] / \
            sum(general_polarity.values())
    return scaled_polarity

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


# In[84]:


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

today_x = btc_df.tail(1)
# In[85]:

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

# In[86]:


def generateFeatures(features):
    complete_features = []
    for feature in features:
        complete_features.append(feature)
        for i in range(LOOK_BACK_DAYS):
            complete_features.append(feature + "_Past_" + str(i + 1))
    return complete_features


features = generateFeatures(['Close', 'Open'])
today_x = today_x[features]
polarities = today_news()
today_x["negative"] = [polarities['neg']]
today_x["neutral"] = [polarities['neu']]
today_x["positive"] = [polarities['pos']]
features.append('MonthTrend')

# selecting the important feature open close
btc = btc_df[features]
btc

# In[87]:


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
btc = btc.dropna()
btc


# In[88]:


Y = btc['MonthTrend']
le = LabelEncoder()
Y = le.fit_transform(Y)
X = btc.drop(['MonthTrend'], axis=1)
X = StandardScaler().fit_transform(X)
today_x = StandardScaler().fit_transform(today_x)
# Create training and testing datasets that are appropriate for time series data
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, random_state=2)
train_size_perc = 0.7
n_time, n_features = X.shape
train_size = np.int16(np.round(train_size_perc * n_time))
X_train, Y_train = X[:X.shape[0]], Y[:X.shape[0]]


# In[89]:


# decision tree


# logistic regression
model_regression = LogisticRegression()
param_grid = {
    'C': [1, 2, 4, 6, 8, 10],
    'penalty': ["l1", "l2"],
    'fit_intercept': [True, False],
    'class_weight': ["balanced", None],
    'warm_start': [True, False]}
regression_grid = GridSearchCV(model_regression, param_grid, scoring="roc_auc", cv=TimeSeriesSplit(
    max_train_size=None, n_splits=10), verbose=1, n_jobs=6)
regression_grid.fit(X_train, Y_train)
model_regresssion = regression_grid.best_estimator_
#regression_error_rate = 1 - model_regresssion.score(X_test, Y_test)
# regression_error_rate


# In[105]:


regression_prediction = model_regresssion.predict(today_x)
if regression_prediction[0] == 0:
    print("price of BTC predicted to go down tomorrow")
else:
    print("price of BTC predicted to go up tomorrow")
