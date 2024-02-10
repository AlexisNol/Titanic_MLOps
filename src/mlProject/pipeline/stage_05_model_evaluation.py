from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.model_evaluation import ModelEvaluation

from mlProject import logger
from pathlib import Path

from mlProject import logger

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        print('nous sommes dans la classe stage 5 model evaluation (old version)')
        config=ConfigurationManager()
        model_evaluation_config=config.get_model_evaluation_config()
        model_trainer=ModelEvaluation(config=model_evaluation_config)
        model_trainer.log_into_mlflow()
