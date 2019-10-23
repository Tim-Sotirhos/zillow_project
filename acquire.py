import pandas as pd
import numpy as np

import env

def get_db_url(database_name):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{database_name}'

def get_data_from_mysql():
    query = '''
    SELECT customer_id, monthly_charges, tenure, total_charges
    FROM customers WHERE contract_type_id = 3
    ORDER BY total_charges DESC;
    '''

    df = pd.read_sql(query, get_db_url('telco_churn'))
    return df

def clean_data(df):
    df = df[df.total_charges != ' ']
    df.total_charges = df.total_charges.astype(float)
    return df

def wrangle_telco():
    df = get_data_from_mysql()
    df = clean_data(df)
    return df

def get_titanic_data():
    query = '''
    SELECT * FROM passengers
    '''

    df = pd.read_sql(query, get_db_url('titanic_db'))
    return df

def get_iris_data():
    query = '''
    SELECT * FROM measurements 
    JOIN species USING(species_id)
    '''
    df = pd.read_sql(query,get_db_url('iris_db'))
    return df