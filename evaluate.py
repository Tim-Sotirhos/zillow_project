# Evaluating Regression Models Functions

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from math import sqrt
import matplotlib.pyplot as plt
from pydataset import data

# import data and explore
tips = data('tips')
tips = tips[["total_bill", "tip"]]
tips.head()
tips.shape
tips.describe()
tips.info()
type(tips)
tips.isnull().sum()

# create dataframes variables x and y
x = pd.DataFrame(tips['total_bill'])
y = pd.DataFrame(tips['tip'])

# fit the model to your data, where x = total_bill and y = tip 
regr = ols('y ~ x', data = tips).fit()

# compute yhat, the predictions of tip using total_bill
tips['yhat'] = regr.predict(x)
tips.describe()
tips.head()
regr.summary()

# compute residuals
tips['residual'] = tips['yhat'] - tips['tip']
tips.head()

# square of each residual
tips['residual^2'] = tips.residual ** 2

# 4.) Function that takes the feature, the target, and the dataframe as input and returns a residual plot.

def plot_residuals(x, y, data_frame):
    plt.title(r'Baseline Residuals', fontsize=12, color='white')
    plt.ylabel(r'$\hat{y}-y$')
    plt.xlabel('Tip')
    return sns.residplot(x, y, tips)
    
plot_residuals(x,y,tips)

# 5.) Function that takes in y and yhat,
# returns the sum of squared errors (SSE), 
# explained sum of squares (ESS), 
# total sum of squares (TSS), 
# mean squared error (MSE), 
# and root mean squared error (RMSE).

def regression_errors(y, yhat, data_frame):
    # SSE - sum of squared errors using MSE * len()
    SSE = mean_squared_error(y, tips.yhat) * len(tips)
    # MSE - mean of squared errors
    MSE = mean_squared_error(y, tips.yhat)
    # ESS - explained sum of squares
    ESS = sum((tips.yhat - tips.tip.mean()) ** 2)
    # TSS - total sum of squares
    TSS = SSE + ESS
    # RMSE - root mean squared error
    RMSE = sqrt(MSE)
    print("SSE: ", SSE, "EES: ", ESS, "TSS: ", TSS, "MSE: ", MSE, "RMSE: ", RMSE)
    return SSE, ESS, TSS, MSE, RMSE

regression_errors(y, tips.yhat, tips)

# 6.) Function that takes in your target, y, 
# computes the SSE, MSE & RMSE when yhat is equal to the mean of all y, 
# and returns the error values (SSE, MSE, and RMSE).

# copy dataframe
tips_baseline = tips[['total_bill','tip']]
tips_baseline.head()

# compute the overall mean of the y values and add to 'yhat' as our prediction
tips_baseline['yhat'] = tips_baseline['tip'].mean()
tips_baseline.head()

# compute the difference between y and yhat
tips_baseline['residual'] = tips_baseline['yhat'] - tips_baseline['tip']

# square that delta
tips_baseline['residual^2'] = tips_baseline['residual'] ** 2

tips_baseline.head(3)

def baseline_mean_errors(y):
    SSE = mean_squared_error(tips_baseline.tip, tips_baseline.yhat) * len(tips_baseline)
    MSE = mean_squared_error(tips_baseline.tip, tips_baseline.yhat)
    RMSE = sqrt(MSE)
    print("SSE: ", SSE, "MSE: ", MSE, "RMSE: ", RMSE)
    return SSE, MSE, RMSE

baseline_mean_errors(tips_baseline.tip)

# 7.) function that returns true if your model performs better than the baseline, otherwise false.

def better_than_baseline(data_frame, data_frame_baseline):
    SSE = mean_squared_error(tips.tip, tips.yhat) * len(tips)
    SSE_bl = mean_squared_error(tips_baseline.tip, tips_baseline.yhat) * len(tips_baseline)
    MSE = mean_squared_error(tips.tip, tips.yhat)
    MSE_bl = mean_squared_error(tips_baseline.tip, tips_baseline.yhat)
    if SSE < SSE_bl and MSE < MSE_bl:
        return True

better_than_baseline(tips, tips_baseline)

#  8.) Function that takes the ols model as input and returns the amount of variance explained in your model, 
# and the value telling you whether the correlation between the model and the tip value are statistically significant.

def model_significance(ols_model):
    r2 = ols_model.rsquared
    f_pval = ols_model.f_pvalue
    if f_pval < .05:
        return f'An R^2 of: {round(r2, 3)} and an F Statistic of: {round(f_pval, 4)} explains that the correlation between the model and the tip value is significantly significant'
    else:
        return f'The correlation between the model and the tip value is not significantly significant'

model_significance(regr)