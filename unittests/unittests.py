"""
Script with unit tests for different methods of train.py and run.py.
"""
# Importing required libraries
import unittest
import pandas as pd
import os
import sys
import json

# Add the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = "settings.json" 

# Importing custom packages
from training.train import IrisNN, DataProcessor, Training
from inference.run import get_model, prepare_inference_dataloader, \
    predict_results, store_results 


# Defining classes and methods for different functionalities
class TestModelTraining(unittest.TestCase):
    """
    This class contains unit tests for model training functionalities.
    It tests the training process, model evaluation, and model parameter updates.
    """
    def setUp(self):
        """Sets up mock data and model for training tests."""
        self.X_mock = pd.DataFrame({
            'feature1': [0.1, 0.2, 0.3],
            'feature2': [1, 2, 3],
            'feature3': [0.4, 0.5, 0.6],
            'feature4': [4, 5, 6]
        })
        self.y_mock = pd.Series([0, 1, 0])
        self.data_proc = DataProcessor()
        self.train_loader = self.data_proc.convert_to_dataloader(self.X_mock, 
                                                                 self.y_mock, 
                                                                 batch_size=1, 
                                                                 shuffle=False)
        self.training = Training()
        self.model = IrisNN()
        self.trained_model = self.training.train(self.model, self.train_loader, 
                                                 self.train_loader, epochs=1, lr=0.001)

    def test_train_model(self):
        """Tests if the model is successfully trained with default parameters."""
        for param in self.trained_model.parameters():
            self.assertTrue(param.requires_grad)

    def test_evaluate(self):
        """Tests the evaluation function on trained model."""
        accuracy, class_report = self.training.evaluate(self.trained_model, 
                                                        self.train_loader)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(class_report, str)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


class TestDataProcessor(unittest.TestCase):
    """
    This class contains unit tests for data processing functionalities.
    It tests data extraction, data splitting, data preparation, and DataLoader conversion.
    """
    @classmethod
    def setUpClass(cls):
        """Sets up the class for data processing tests."""
        with open(CONF_FILE, "r") as file:
            cls.conf = json.load(file)
        cls.data_proc = DataProcessor()
        cls.train_path = os.path.join(cls.conf['general']['data_dir'], 
                                      cls.conf['train']['table_name'])
        cls.test_size = cls.conf['train']['test_size']
        cls.batch_size = 1  

    def test_data_extraction(self):
        """Tests data extraction from a CSV file."""
        df = self.data_proc.data_extraction(self.train_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.shape[0], 0)

    def test_data_split(self):
        """Tests splitting data into training and testing sets."""
        df = self.data_proc.data_extraction(self.train_path)
        X_train, X_test, y_train, y_test = self.data_proc.data_split(df, self.test_size)
        self.assertEqual(len(X_train) + len(X_test), len(df))

    def test_prepare_data(self):
        """Tests preparation of data for model training."""
        train_loader, test_loader = self.data_proc.prepare_data()
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)

    def test_convert_to_dataloader(self):
        """Tests conversion of data to DataLoader with and without labels."""
        X_mock = pd.DataFrame({'feature1': [0.1, 0.2], 'feature2': [1, 2]})
        y_mock = pd.Series([0, 1])

        loader_with_labels = self.data_proc.convert_to_dataloader(X_mock, y_mock, 
                                                                  batch_size=1, 
                                                                  shuffle=False)
        for inputs, targets in loader_with_labels:
            self.assertEqual(inputs.shape[1], X_mock.shape[1])
            self.assertEqual(targets.shape[0], 1)

        loader_without_labels = self.data_proc.convert_to_dataloader(X_mock, 
                                                                     batch_size=1, 
                                                                     shuffle=False)
        for inputs in loader_without_labels:
            self.assertEqual(inputs[0].shape[1], X_mock.shape[1])


class TestInference(unittest.TestCase):
    """
    This class contains unit tests for inference functionalities.
    It tests the model loading, DataLoader preparation for inference, prediction results,
    and storing of inference results.
    """
    @classmethod
    def setUpClass(cls):
        """Sets up the class for inference tests."""
        with open(CONF_FILE, "r") as file:
            cls.conf = json.load(file)
        cls.infer_data = pd.DataFrame({
            'feature1': [0.1, 0.2, 0.3],
            'feature2': [1, 2, 3],
            'feature3': [0.4, 0.5, 0.6],
            'feature4': [4, 5, 6]
        })
    
    def test_get_model(self):
        """Tests retrieval of the trained model."""
        model = get_model()
        self.assertIsInstance(model, IrisNN)

    def test_prepare_inference_dataloader(self):
        """Tests preparation of DataLoader for inference."""
        infer_loader = prepare_inference_dataloader(self.infer_data)
        for inputs in infer_loader:
            self.assertEqual(inputs[0].shape[1], self.infer_data.shape[1])

    def test_predict_results(self):
        """Tests the prediction results from inference."""
        model = get_model()
        infer_loader = prepare_inference_dataloader(self.infer_data)
        infer_data_copy = self.infer_data.copy()
        results = predict_results(model, infer_data_copy, infer_loader)
        self.assertIn('results', results.columns)

    def test_store_results(self):
        """Tests storing of inference results in a CSV file."""
        results = self.infer_data.copy()
        results['results'] = [0, 1, 0]
        store_results(results, results_name='unittest_results.csv')
        saved_path = os.path.join(self.conf['general']['results_dir'], 
                                  'unittest_results.csv')
        self.assertTrue(os.path.exists(saved_path))


if __name__ == '__main__':
    unittest.main()