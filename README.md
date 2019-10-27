Zillow Project Information

Important:

1. Produce a model that is configured to predict the values of single unit properties that the tax district assesses using the property data from those whose last transaction was during the "hot months" (in terms of real estate demand) of May and June in 2017.

2. Provide the distribution of tax rates by county. Include the distribution of tax rates for each county inorder to see how much they vary within the properties in the county and the rates the bulk of the properties sit around.  Visualize the findings on one chart.

3. For the first iteration of the Zestimate model, use only square feet of the home, number of bedrooms, and number of bathrooms to estimate the properties assessed value. The result will be the mvp (minimally viable product).

Pipeline and special instructions:

ACQIUIRE - Obtain data from SQL zillow database. 

PREP - Only include "single family residences" with a vaild "bedroom count" and "bathroom count" of both greater than zero.

SPLIT & SCALE - Create an X and y dataframe. Then split the data into train (X, y) and test (X, y) with a ratio of .80 for train and .20 for test. Also used a random_state equal to 123. Use a Standard Scaler to scale entire data set (split: train and test). 

DATA EXPLORATION - Create a correlation heatmap to visualize the established dependencies among all variables (independent and target). Preform 1 t-test using ordinary least squares (OLS) regression anaysis results.

MODELING & EVALUATION - Create a baseline model using only the target variable's mean. Create one linear regression model using three selected independent variables. Compare each model's evaluation  metrics to determine which provides a better fitting line plot.  Select best model and use test data. 

