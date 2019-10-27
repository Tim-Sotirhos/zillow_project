import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,explained_variance_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std


#import our scripts that do data science workflow
import wrangle
import split_scale
import evaluate
import features

# Create dataframe of only monthly_charges, tenure, and total_charges

df = df[['monthly_charges', 'tenure', 'total_charges']]
df.info()

# Create data frames of X_train, x_test, y_train, y_test.

train, test = split_scale.split_my_data(df)
X_train = train[['monthly_charges','tenure']]
y_train = train[['total_charges']]

X_test = test[['monthly_charges', 'tenure']]
y_test = test[['total_charges']]

X_train, y_train, X_test, y_test

# Scale data using standard scaler

scaler, train_scaled, test_scaled = split_scale.standard_scaler(df)
scaler, train_scaled, test_scaled

# Create X_train_scaled, y_train_scaled, X_test_scaled, and y_test_scaled

X_train_scaled = train_scaled[['monthly_charges','tenure']]
y_train_scaled = train_scaled[['total_charges']]

X_test_scaled = test_scaled[['monthly_charges', 'tenure']]
y_test_scaled = test_scaled[['total_charges']]

X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled

def modeling_function(X_train,X_test,y_train,y_test):
    predictions_train=pd.DataFrame({'actual':y_train.taxvaluedollarcnt}).reset_index(drop=True)
    predictions_test=pd.DataFrame({'actual':y_test.taxvaluedollarcnt}).reset_index(drop=True)
    #model 1
    lm1=LinearRegression()
    lm1.fit(X_train,y_train)
    lm1_predictions=lm1.predict(X_train)
    predictions_train['lm1']=lm1_predictions

    #model 2
    lm2=LinearRegression()
    lm2.fit(X_test,y_test)
    lm2_predictions=lm2.predict(X_test)
    predictions_test['lm2']=lm2_predictions

    #model 3 - baseline -train
    lm3_predictions = np.array([y_train.mean()[0]]*len(y_train))
    predictions_train['baseline'] = lm3_predictions

    #model 3 - baseline - test
    lm4_predictions = np.array([y_test.mean()[0]]*len(y_test))
    predictions_test['baseline'] = lm4_predictions

def plot_residuals(x, y):
    '''
    Plots the residuals of a model that uses x to predict y. Note that we don't
    need to make any predictions ourselves here, seaborn will create the model
    and predictions for us under the hood with the `residplot` function.
    '''
    return sns.residplot(x, y)

def plot_regression(x,y):
    res = sm.OLS(y, x).fit()
    prstd, iv_l, iv_u = wls_prediction_std(res)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(x, y, 'o', label="data")
    #ax.plot(x, y, 'b-', label="True")
    ax.plot(x, res.fittedvalues, 'r--.', label="OLS")
    ax.plot(x, iv_u, 'r--')
    ax.plot(x, iv_l, 'r--')
    ax.legend(loc='best');
    plt.show()