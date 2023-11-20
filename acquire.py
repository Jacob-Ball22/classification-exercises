import pandas as pd
import env
import os
#1. Make a function named get_titanic_data that returns the Titanic data from the codeup data science 
# database as a pandas data frame. 
def get_titanic_data():
    filename = 'titanic.csv'
    if os.path.exists(filename):
        print('this file exists, reading from csv')
        #read from csv
        df = pd.read_csv(filename)
    else:
        print('this file doesnt exist, reading from sql and saving to csv')
        #read from sql
        url = env.get_db_url('titanic_db')
        df = pd.read_sql('select * from passengers', url)
        #save to csv
        df.to_csv(filename)
    return df

#2. Make a function named get_iris_data that returns the data from the iris_db on the codeup data science database as a pandas data frame. 
# The returned data frame should include the actual name of the species in addition to the species_ids.
def get_iris_data():
    filename = 'iris.csv'
    if os.path.exists(filename):
        print('this file exists, reading from csv')
        #read from csv
        df = pd.read_csv(filename)
    else:
        print('this file doesnt exist, reading from sql and saving to csv')
        #read from sql
        url = env.get_db_url('iris_db')
        df = pd.read_sql('select * from measurements join species using (species_id)', url)
        #save to csv
        df.to_csv(filename)
    return df

#3. Make a function named get_telco_data that returns the data from the telco_churn database in SQL. 
# In your SQL, be sure to join contract_types, internet_service_types, payment_types 
# tables with the customers table, so that the resulting dataframe contains all the 
# contract, payment, and internet service options.
def get_telco_data():
    filename = 'telco.csv'
    if os.path.exists(filename):
        print('this file exists, reading from csv')
        #read from csv
        df = pd.read_csv(filename)
    else:
        print('this file doesnt exist, reading from sql and saving to csv')
        #read from sql
        url = env.get_db_url('telco_churn')
        df = pd.read_sql('''select * from customers 
    join contract_types ct
		using (contract_type_id)
	join internet_service_types ist
		using (internet_service_type_id)
	join payment_types pt
		using (payment_type_id)''', url)
        #save to csv
        df.to_csv(filename)
    return df