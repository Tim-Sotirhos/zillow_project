# Scaling Numeric Data exercise

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import wrangle
import env
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer,RobustScaler,MinMaxScaler

df = wrangle.wrangle_telco()
df.info()
df.describe()
df.head()
df.drop(columns = ['customer_id'], inplace = True)
df.head()

# 1.) split_my_data(X, y, train_pct)

def split_my_data(data, train_ratio = .80, seed = 123):
    train, test = train_test_split(data, train_size = train_ratio, random_state = seed)
    return train, test

# 2.) standard_scaler()

def standard_scaler(train, test):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    train_scaled_data = scaler.transform(train)
    test_scaled_data = scaler.transform(test)
    train_scaled = pd.DataFrame(train_scaled_data, columns=train.columns).set_index([train.index])
    test_scaled = pd.DataFrame(test_scaled_data, columns=test.columns).set_index([test.index])
    return scaler, train_scaled, test_scaled

# 3.) scale_inverse()
def scale_inverse(scaler, train_scaled, test_scaled):
    train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns).set_index([train.index])
    test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns).set_index([test.index])
    return train_unscaled, test_unscaled

# 4.) uniform_scaler()

def uniform_scaler(train, test):
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

# 5.) gaussian_scaler()

def gaussian_scaler(train, test):
    scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

# 6.) min_max_scaler()
def min_max_scaler(train, test):
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

# 7.) iqr_robust_scaler()

def iqr_robust_scaler(train, test):
    scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled