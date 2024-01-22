"""
This script defines the neural network model, prepares the dataloaders. 
Runs the training, evaluates model on the test set, and saves the model.
"""
# Importing required libraries
import argparse
import os
import sys
import json
import logging
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Ignore specific warnings from scikit-learn
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Add the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = "settings.json" 

# Importing custom packages
from utils import set_seed, get_project_dir, configure_logging

# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initialize parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", 
                    help="Specify inference data file", 
                    default=conf['train']['table_name'])
parser.add_argument("--model_path", 
                    help="Specify the path for the output model")

# Define classes and methods
class DataProcessor():
    """
    Handles the data processing tasks such as loading data, splitting into 
    train/test sets, and converting data to DataLoader format.
    """
    def __init__(self) -> None:
        pass

    def prepare_data(self) -> DataLoader:
        """Converts a single train DataFrame to train and test DataLoaders"""
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        logging.info(f"Train dataset contains {df.shape[1]} columns and {df.shape[0]} rows")
        X_train, X_test, y_train, y_test = self.data_split(df)
        logging.info(f"Model to be trained with {X_train.shape[0]}" 
                     f"observations and tested on {X_test.shape[0]} observations")
        train_loader = self.convert_to_dataloader(X_train, y_train, shuffle=True)
        test_loader = self.convert_to_dataloader(X_test, y_test, shuffle=False)
        return train_loader, test_loader

    def data_extraction(self, path: str) -> pd.DataFrame:
        """Loads a .csv file and converts it DataFrame."""
        if not os.path.isfile(path):
            raise FileNotFoundError("The specified dataset does not exist at "
                                    f"{path}. Check the file path.")
        try:
            logging.info(f"Loading dataset from {path}...")
            df = pd.read_csv(path)
            logging.info("Dataset loaded. It contains "
                         f"{df.shape[1]} columns and {df.shape[0]} rows.")
            return df
        except Exception as e:
            logging.error(f"An error occurred while loading data from {path}: {e}")
            sys.exit(1)
    
    def data_split(self, df: pd.DataFrame, test_size: float = conf['train']['test_size']) -> tuple:
        """Splits data into test and train."""
        logging.info("Splitting data into training and test sets...")
        return train_test_split(df.drop("target", axis=1), df["target"], 
                                test_size=test_size, stratify=df["target"],
                                random_state=conf['general']['random_state']) 

    def convert_to_dataloader(self, X: pd.DataFrame, y: pd.DataFrame = None, 
                              batch_size: int = conf['train']['batch_size'], 
                              shuffle: bool = False) -> DataLoader:
        """Converts DataFrame to DataLoader for training or inference."""
        tensor_x = torch.tensor(X.values, dtype=torch.float32)
        if y is not None:
            tensor_y = torch.tensor(y.values, dtype=torch.long)
            dataset = TensorDataset(tensor_x, tensor_y)
        else:
            dataset = TensorDataset(tensor_x)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, 
                          worker_init_fn=conf['general']['random_state'])


class IrisNNBase(nn.Module):
    """
    Base neural network model class, defining the layer structure.
    """
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)      
        return x


class IrisNN(IrisNNBase):
    """
    Neural network model with 3 hidden layers, extending the base model.
    """ 
    def __init__(self):
        super().__init__(conf['train']['nn_architecture']) 


class Training():
    """
    Manages the training process including running training, evaluating 
    the model, and saving the trained model.
    """
    def __init__(self) -> None:
        pass

    def run_training(self, train_loader: DataLoader, test_loader: DataLoader, 
                     epochs: int = conf['train']['epochs'], 
                     lr: float = conf['train']['lr']) -> None:
        """Runs the model training process including evaluation and saving."""
        logging.info("Running training...")
        start_time = time.time() 
        model = IrisNN()
        trained_model = self.train(model, train_loader, test_loader, epochs, lr)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")
        _, class_report = self.evaluate(trained_model, test_loader)
        logging.info(f"Classification Report on Test Set:\n{class_report}")
        self.save(model=trained_model)  

    def evaluate(self, model: IrisNN, test_loader: DataLoader) -> tuple:
        """Evaluates the trained model on a test set."""
        if test_loader is None or len(test_loader) == 0:
            raise ValueError("No test dataset provided or test_loader is empty.")
        model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                all_predictions.append(predictions)
                all_targets.append(targets)
        all_predictions = torch.cat(all_predictions).cpu().numpy()
        all_targets = torch.cat(all_targets).cpu().numpy()
        accuracy = accuracy_score(all_targets, all_predictions)
        class_report = classification_report(all_targets, all_predictions)
        return accuracy, class_report
    
    def train(self, model: IrisNN, train_loader: DataLoader, test_loader: DataLoader, 
              epochs: int = conf['train']['epochs'], lr: float = conf['train']['lr']) -> IrisNN:
        """Trains the model using the provided dataloaders."""
        set_seed(conf['general']['random_state'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        best_accuracy = 0.0
        best_model = None
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            current_test_accuracy, _ = self.evaluate(model, test_loader)
            if current_test_accuracy > best_accuracy:
                best_accuracy = current_test_accuracy
                best_model = deepcopy(model)
        return best_model if best_model is not None else model
    
    def save(self, model: IrisNN, model_name: str = conf['train']['model_name']) -> None:
        """Saves the trained model to the specified path."""
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        # Use the model name from settings.json
        final_path = os.path.join(MODEL_DIR, model_name)
        torch.save(model.state_dict(), final_path)
        logging.info(f"Model saved to {final_path}")


def main():
    """Main function"""
    configure_logging()

    data_proc = DataProcessor()
    tr = Training()

    train_loader, test_loader = data_proc.prepare_data()
    tr.run_training(train_loader, test_loader)

# Executing the script
if __name__ == "__main__":
    main()
