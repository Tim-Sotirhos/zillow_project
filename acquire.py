import pandas as pd
import numpy as np
import env

def get_db_url(database_name):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{database_name}'

def get_zillow_data():
    query = '''
    SELECT bedroomcnt as bedroom_count, 
    bathroomcnt as bathroom_count, 
    calculatedfinishedsquarefeet as square_feet,
    taxvaluedollarcnt as property_value,
    taxamount as tax_paid,
    (taxamount / taxvaluedollarcnt) * 100 as tax_rate,
    proptype.propertylandusedesc as property_type, 
    fips as county_code
    FROM properties_2017 AS property
    JOIN predictions_2017 AS pred USING(parcelid)
    JOIN propertylandusetype AS proptype USING(propertylandusetypeid)
    WHERE (transactiondate >= '2017-05-01' AND transactiondate <= '2017-06-30')
    AND (propertylandusedesc = 'Single Family Residential')
    AND property.bedroomcnt > 0
    AND property.bathroomcnt > 0
    AND property.taxvaluedollarcnt > 0
    AND property.lotsizesquarefeet > 0
    ORDER BY fips;
    '''

    df = pd.read_sql(query, get_db_url('zillow'))
    return df    

df = get_zillow_data()

def clean_data(df):
    df = df.dropna()
    return df

def wrangle_zillow():
    df = get_zillow_data()
    df = clean_data(df)
    return df