from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.data_ingestion import DataIngestion
from src.mlProject import logger




class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        print("step ok")
        config = ConfigurationManager() 
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion= DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()



