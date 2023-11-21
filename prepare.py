import pandas as pd

import env
import os
import acquire

def prep_iris():
    """
    Gets iris data from SQL or .csv
    Lowercases all columns and replaces all periods with underscores
    Removes 'species_id' and 'measurement_id'
    Renames 'species_name' to 'species'
    """
    iris_df = acquire.get_iris_data()
    iris_df.columns = [col.lower().replace('.', '_') for col in iris_df.columns]
    iris_df = iris_df.drop(columns=['species_id','measurement_id'])
    iris_df = iris_df.rename(columns={'species_name':'species'})
    
    return iris_df

def prep_titanic():
    """
    Gets titanic data from SQL or .csv
    Drops columns 'embarked','class','age', and 'deck'
    Changes 'pclass' to an object type
    Fills in null values in 'embark_town' with 'Southampton'
    """
    df = acquire.get_titanic_data()
    df = df.drop(columns=['embarked','class','age','deck'])
    df.pclass = df.pclass.astype(object)
    df.embark_town = df.embark_town.fillna('Southampton')
    
    return df

def prep_telco():
    """
    Gets telco data from SQL or .csv
    Drops columns 'payment_type_id', 'internet_service_type_id', and 'contract_type_id'
    Removes all rows with null values in 'internet_service_type'
    """
    telco_df = acquire.get_telco_data()
    telco_df = telco_df.drop(columns=['payment_type_id','internet_service_type_id','contract_type_id'])
    telco_df.internet_service_type = telco_df.internet_service_type.notnull()
    
    return telco_df

def split_data(df, col):
    """
    Recieves dataframe as 'df' and target variable to stratify as 'col'
    First split does a 60% train and 40% validate
    Second split uses the 40% validate to make 50% validate and 50% test
    """
    #first split
    train, validate_test = train_test_split(df, #send in initial df
                train_size = 0.60, #size of the train df, and the test size will default to 1-train_size
                random_state = 123, #set any number here for consistency
                stratify = df[col] #we should stratify on our target variable
                )
    
    #second split
    validate, test = train_test_split(validate_test, #we are spliting the 40% df we just made
                test_size = 0.50, #split 50/50
                random_state = 123, #gotta send in a random seed
                stratify = validate_test[col] #still got to stratify
                )
    
    return train, validate, test