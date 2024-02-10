import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import ElasticNet

import joblib


from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config=config
    

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        print("ok")
        train_x= train_data.drop([self.config.target_column],axis=1)
        test_x= test_data.drop([self.config.target_column],axis=1)
        train_y= train_data[[self.config.target_column]]
        test_y= test_data[[self.config.target_column]]
