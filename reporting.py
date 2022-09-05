import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['test_data_path']) 
img_save_path = os.path.join(config['output_model_path'])


##############Function for reporting
def evaluate_model(is_redeploy=False):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    y_true, y_pred = model_predictions(os.path.join(dataset_csv_path, "test_data.csv"))

    cm = confusion_matrix(y_true, y_pred)

    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_true.unique())

    cm_disp.plot()

    if not is_redeploy:
        plt.savefig(os.path.join(img_save_path, "confusion_matrix.png"))
    else:
        plt.savefig(os.path.join(img_save_path, "confusion_matrix_2.png"))


if __name__ == '__main__':
    evaluate_model()
