"""
Script loads the latest trained model, data for inference and predicts results.
"""
# Importing required libraries
import argparse
import json
import logging
import os
import sys
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Add the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = "settings.json" 

# Importing custom packages
from utils import get_project_dir, configure_logging
from training.train import IrisNN, DataProcessor

# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initialize parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")

# Define methods
def get_model(model_name: str = conf['train']['model_name']) -> IrisNN:
    """Loads and returns a trained model."""
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}." 
                                "Please train the model first.")
    try:
        model = IrisNN()  
        model.load_state_dict(torch.load(model_path))
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"An error occurred while loading the model from {model_path}: {e}")
        sys.exit(1)

def prepare_inference_dataloader(infer_data: pd.DataFrame) -> DataLoader:
    """Convert inference dataframe to tensor dataloader."""
    logging.info(f"Preparing dataloader for inference...")
    infer_loader = DataProcessor().convert_to_dataloader(X=infer_data)
    return infer_loader

def predict_results(model: IrisNN, infer_data: pd.DataFrame, 
                    infer_loader: DataLoader) -> pd.DataFrame:
    """Predict results and join it with the inderence dataframe."""
    logging.info("Running inference...")
    start_time = time.time() 
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in infer_loader:
            outputs = model(inputs[0])
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    infer_data["results"] = predictions
    end_time = time.time()
    logging.info(f"Inference completed in {end_time - start_time} seconds.")
    return infer_data

def store_results(results: pd.DataFrame, 
                  results_name: str = conf['inference']['results_name']) -> None:
    """Store the prediction results in 'results' directory."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    path = os.path.join(RESULTS_DIR, results_name)
    results.to_csv(path, index=False)
    logging.info(f"Results saved to {path}")


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()
    model = get_model()
    infer_file = args.infer_file
    infer_data = DataProcessor().data_extraction(os.path.join(DATA_DIR, infer_file))
    infer_loader = prepare_inference_dataloader(infer_data)
    results = predict_results(model, infer_data, infer_loader)
    store_results(results)
    logging.info(f"Prediction results: \n{results}")

# Executing the script
if __name__ == "__main__":
    main()