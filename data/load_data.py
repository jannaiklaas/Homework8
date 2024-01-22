"""
This script imports the Iris dataset. It performs initial data preprocessing.
The data is split into training and inference sets, scaled, and saved locally as .csv files.
"""

# Importing required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logging
import os
import json

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(ROOT_DIR)
CONF_FILE = os.path.join(REPO_DIR, "settings.json")

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
TRAIN_PATH = os.path.join(ROOT_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(ROOT_DIR, conf['inference']['inp_table_name'])

# Define methods
def load_and_preprocess_data():
    """Loads, splits, and scales the Iris dataset from sklearn."""
    logger.info("Importing the Iris dataset...")
    X = load_iris().data
    y = load_iris().target
    logger.info("Splitting the dataset into training and inference...")
    X_train, X_infer, y_train, y_infer = train_test_split(X, y, 
                                                          test_size = conf['general']['infer_size'], 
                                                          stratify = y, 
                                                          random_state = conf['general']['random_state'])
    logger.info("Scaling the features in the datasets...")
    scaler = MinMaxScaler().fit(X_train)  
    X_train = scaler.transform(X_train)
    X_infer = scaler.transform(X_infer)
    return X_train, X_infer, y_train, y_infer

def save_data(X_train: pd.DataFrame, X_infer: pd.DataFrame, 
              y_train: pd.DataFrame, feature_names: list):
    """Stores train and inference datasets."""
    logger.info("Train dataset will contain both features and labels.")
    train_data = pd.DataFrame(X_train, columns = feature_names)
    train_data['target'] = y_train
    train_data.to_csv(TRAIN_PATH, index = False)
    logger.info("Inference dataset will contain only features without labels.")
    infer_data = pd.DataFrame(X_infer, columns = feature_names)
    infer_data.to_csv(INFERENCE_PATH, index = False)
    logger.info("Datasets have been saved.")

# Executing the script
if __name__ == "__main__":
    X_train, X_infer, y_train, y_infer = load_and_preprocess_data()
    save_data(X_train, X_infer, y_train, load_iris().feature_names)
