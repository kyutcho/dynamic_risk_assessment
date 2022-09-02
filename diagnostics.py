
import pandas as pd
import numpy as np
import timeit
import pickle
import os
import json
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

##################Function to get model predictions
def model_predictions(test_df: pd.DataFrame):
    #read the deployed model and a test dataset, calculate predictions
    X_test = test_df
    y_test = X_test["exited"]

    with open(os.path.join(prod_deployment_path, "trained_model.pkl"), 'rb') as f:
        lr = pickle.load(f)

    y_pred = lr.predict(X_test)

    assert X_test.shape[0] == len(y_pred)

    #return value should be a list containing all predictions
    return y_pred

##################Function to get summary statistics
def dataframe_summary(df: pd.DataFrame):
    #calculate summary statistics here
    stats = df.describe(include=np.number).loc[["mean", "50%", "std"], :]
    stats_dict = stats.to_dict(orient='list')

    #return value should be a dicionary of lists containing all summary statistics
    return stats_dict
    

##################Function to get percent of missing data
def dataframe_missing_data(df: pd.DataFrame):
    na_pct = round((df.isnull().sum() / len(df) * 100), 2)

    return na_pct


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_time = timeit.default_timer()
    os.system('python3 training.py')
    time1 = timeit.default_timer() - start_time
    
    start_time = timeit.default_timer()
    os.system('python3 ingestion.py')
    time2 = timeit.default_timer() - start_time

    times_lst = [time1, time2]

    #return a list of 2 timing values in seconds
    return times_lst


##################Function to check dependencies
def outdated_packages_list():
    #get a list
    pip_list = subprocess.check_output(['pip', 'list'])
    with open('pip_list.txt', 'wb') as f:
       f.write(pip_list)


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    dataframe_missing_data()
    execution_time()
    outdated_packages_list()





    
