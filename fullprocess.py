import os
import json
import glob
import logging
from ingestion import *
from training import *
from scoring import *
from deployment import *
from diagnostics import *
from reporting import *
from apicalls import *

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

with open('config.json', 'r') as f:
    config = json.load(f)

deployment_path = os.path.join(config["prod_deployment_path"])
input_data_path = os.path.join(config["input_folder_path"])
data_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config['output_model_path'])

##################Check and read new data
def check_new_files():
    # Read ingestedfiles.txt
    curr_data_lst = []
    with open(os.path.join(deployment_path, "ingested_files.txt")) as f:
        for line in f:
            curr_data_lst.append(line.strip())

    # Check new data in input_folder_path
    os.chdir(input_data_path)
    new_data_lst = glob.glob(f"*.csv")
    os.chdir('..')

    # Determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    to_ingest_data_lst = set(new_data_lst).difference(set(curr_data_lst))

    return len(to_ingest_data_lst) != 0


##################Checking for model drift
def check_model_drift():
    with open(os.path.join(deployment_path, 'latest_score.txt'), 'r') as f:
        old_f1_score = float(f.read())

    # Get new data
    new_data = pd.read_csv(os.path.join(data_csv_path, "final_data.csv"))

    # Re-train model
    train_model()

    # Load new model
    with open(os.path.join(model_path, 'trained_model.pkl'), 'rb') as f:
        new_lr = pickle.load(f)

    new_f1_score = score_model(new_lr, new_data)

    return new_f1_score < old_f1_score


def full_pipeline():
    # Check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    if_new_files = check_new_files()

    # If you found new data, you should proceed. otherwise, do end the process here
    if if_new_files:
        logger.info("New file detected. Files are merged and ingested to new file")
        merge_multiple_dataframe()
    else:
        logger.info("No new file detected, hence do not proceed to model drift check")
        exit()
    
    if_model_drifts = check_model_drift()

    # If you found model drift, you should proceed. otherwise, do end the process here
    if if_model_drifts:
        logger.info("Model drift detected. Redeploy model.")
        with open(os.path.join(model_path, 'trained_model.pkl'), 'rb') as f:
            new_model = pickle.load(f)

        # If you found evidence for model drift, re-deploy model. Also, run diagnostics and reporting for the re-deployed model
        store_model_into_pickle(new_model)
        evaluate_model(is_redeploy=True)
        api_calls(is_redeploy=True)

    else:
        logger.info("No model drift detected, hence do not proceed to re-deployment")
        exit()
    

if __name__ == '__main__':
    full_pipeline()