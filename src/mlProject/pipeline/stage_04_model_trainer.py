from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.model_trainer import ModelTrainer

from mlProject import logger
from pathlib import Path

from mlProject import logger

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        config=ConfigurationManager()
        model_trainer_config=config.get_model_trainer()
        model_trainer=ModelTrainer(config=model_trainer_config)
        model_trainer.train()