# Feature Engineering Exercise

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.feature_selection import SelectKBest, f_regression
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE



import env
import wrangle
import split_scale

# Acquire and prep data
df = wrangle.wrangle_telco()
df.head()
df.info()
df.drop(columns = ['customer_id'], inplace = True)
df.head()

# Explore data
sns.pairplot(data = df)

# split data
train, test = split_scale.split_my_data(data = df, train_ratio = .80, seed = 123)

# Scale

scaler, train_scaled, test_scaled = split_scale.standard_scaler(train, test)

# Seperate into X and y dataframes

X_train = train.drop(columns=['total_charges'])
y_train = train[['total_charges']]

X_test = test.drop(columns=['total_charges'])
y_test = test[['total_charges']]

X_train_scaled = train_scaled.drop(columns=['total_charges'])
y_train_scaled = train_scaled[['total_charges']]

X_test_scaled = test_scaled.drop(columns=['total_charges'])
y_test_scaled = test_scaled[['total_charges']]

# 1. Write a function, select_kbest_freg_unscaled(X_train,y_train,k) that takes X_train, y_train and k as input 
# (X_train and y_train should not be scaled!) and returns a list of the top k features

def select_kbest_freg_unscaled(X_train, y_train, k):
    '''
    Takes unscaled data (X_train, y_train) and number of features to select (k) as input
    and returns a list of the top k features
    '''
    f_selector = SelectKBest(f_regression, k).fit(X_train, y_train).get_support()
    f_feature = X_train.loc[:,f_selector].columns.tolist()
    return f_feature

# select_k_best_freg_unscaled(X_train_scaled_data,y_train_scaled_data,2)

# 2. Write a function, select_kbest_freg() that takes X_train, y_train(scaled) and k as input 
# and returns a list of the top k features.

def select_kbest_freg_scaled(X_train, y_train, k):
    '''
    Takes unscaled data (X_train, y_train) and number of features to select (k) as input
    and returns a list of the top k features
    '''
    f_selector = SelectKBest(f_regression, k).fit(X_train, y_train).get_support()
    f_feature = X_train.loc[:,f_selector].columns.tolist()
    return f_feature
select_kbest_freg_scaled(X_train, y_train, 2)
# select_kbest_freg_scaled(X_train_scaled, y_train_scaled, 2)

def select_kbest_freg(X, y, k):
    '''
    dataframe of features (X),  dataframe of the target (y), and number of features to select (k) as input
    and returns a list of the top k features
    '''
    f_selector = SelectKBest(f_regression, k).fit(X, y).get_support()
    f_feature = X.loc[:,f_selector].columns.tolist()
    return f_feature

# 3. Write a function, ols_backward_elimination() that takes X_train and y_train(scaled) as input 
# and returns selected features based on the ols backwards elimation method.

def ols_backward_elimination(X_train, y_train):
    '''
    Takes dataframe of features and dataframe of target variable as input,
    runs OLS, extracts each features p-value, removes the column with the highest p-value
    until there are no features remaining with a p-value > 0.05
    It then returns a list of the names of the selected features
    '''
    cols = list(X_train.columns)

    while (len(cols) > 0):
        # create a new dataframe that we will use to train the model...each time we loop through it will 
        # remove the feature with the highest p-value IF that p-value is greater than 0.05.
        # if there are no p-values > 0.05, then it will only go through the loop one time. 
        X_1 = X_train[cols]
        # fit the Ordinary Least Squares Model
        model = sm.OLS(y_train,X_1).fit()
        # create a series of the pvalues with index as the feature names
        p = pd.Series(model.pvalues)
        # get the max p-value
        pmax = max(p)
        # get the feature that has the max p-value
        feature_with_p_max = p.idxmax()
        # if the max p-value is >0.05, the remove the feature and go back to the start of the loop
        # else break the loop with the column names of all features with a p-value <= 0.05
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break

    selected_features_BE = cols
    return selected_features_BE

# ols_backward_elimination(X_train,y_train)
#ols_backward_elimination(X_train_scaled_data,y_train_scaled_data)

# 4. Write a function, lasso_cv_coef() that takes X_train and y_train as input 
# and returns coefficients for each feature, along with a plot of the the features and their weights.

def lasso_cv_coef(X_train, y_train):
    reg = LassoCV()
    reg.fit(x_train, y_train)
    coef = pd.Series(reg.coef_, index = x_train.columns)
    plot = sns.barplot(x = x_train.columns, y = reg.coef_)
    return coef, plot

# lasso_cv_coef(x_train, y_train)

# 5. Write 3 functions, 
# the first computes the number of optimum features (n) using rfe, 
# the second takes n as input and returns the top n features,
#  and the third takes the list of the top n features as input and returns a new X_train and X_test dataframe with those top features , 
# recursive_feature_elimination() that computes the optimum number of features (n) and returns the top n features.

def optimal_number_of_features(X, y):
    '''discover the optimal number of features, n, using our scaled x and y dataframes, recursive feature
    elimination and linear regression (to test the performance with each number of features).
    We will use the output of this function (the number of features) as input to the next function
    optimal_features, which will then run recursive feature elimination to find the n best features
    '''
    number_of_attributes = X_train.shape[1]
    number_of_features_list=np.arange(1,number_of_attributes)    # len(features_range)

    # set "high score" to be the lowest possible score
    high_score = 0

    # variables to store the feature list and number of features
    number_of_features = 0
    score_list = []
    
    for n in range(len(number_of_features_list)):
        model = LinearRegression()
        rfe = RFE(model,number_of_features_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score):
            high_score = score
            number_of_features = number_of_features_list[n]
    return number_of_features

# optimal_n_features(x_train, y_train)

def optimal_features(X_train, X_test, y_train, number_of_features):
    '''Taking the output of optimal_number_of_features, as n, and use that value to 
    run recursive feature elimination to find the n best features'''
    cols = list(X_train.columns)
    model = LinearRegression()
    
    #Initializing RFE model
    rfe = RFE(model, number_of_features)

    #Transforming data using RFE
    train_rfe = rfe.fit_transform(X_train,y_train)
    test_rfe = rfe.transform(X_test)
    
    #Fitting the data to model
    model.fit(train_rfe, y_train)
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index
    
    X_train_rfe = pd.DataFrame(train_rfe, columns=selected_features_rfe)
    X_test_rfe = pd.DataFrame(test_rfe, columns=selected_features_rfe)
    
    return selected_features_rfe, X_train_rfe, X_test_rfe

def recursive_feature_elimination(X_train, y_train, X_test, y_test):
    '''
    recursive_feature_elimination(X_train, y_train, X_test, y_test)
    RETURNS X_train_optimal, X_test_optimal
    Combines optimal_number_of_features, optimal_features, and 
    create_optimal_dataframe into one single function. Accepts X and y train and 
    test dataframes, returns optimal X train and test dataframes.
    '''

    number_of_features = optimal_number_of_features(X_train, y_train, X_test, y_test)
    selected_features_rfe = optimal_features(X_train, y_train, number_of_features)
    X_train_optimal, X_test_optimal = create_optimal_dataframe(X_train, X_test, selected_features_rfe)

    return X_train_optimal, X_test_optimal