import sys
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformer import DataTransformer
from src.components.model_trainer import ModelTrainer
from src.logger import logging
class TrainingPipeline:
    def __init__(self):
        self.data_path = "Data/spa.txt"
        self.input_tokenizer_path = "artifacts/input_tokenizer.pkl"
        self.output_tokenizer_path = "artifacts/output_tokenizer.pkl"
        self.evaluate_values_path = "artifacts/evaluate_values.pkl"
        self.model_save_path = "artifacts/model.h5"
        self.encoder_save_path = "artifacts/encoder_model.h5"
        self.decoder_save_path = "artifacts/decoder_model.h5"
    def train(self):
        try:
            data_ingestor = DataIngestion(self.data_path)
            data_dict = data_ingestor.split()
            logging.info("data ingestion complete")
            data_transformer = DataTransformer(data_dict,self.input_tokenizer_path,self.output_tokenizer_path, self.evaluate_values_path)
            converted_data_dict = data_transformer.convert()
            logging.info("data transformation complete")
            model_trainer = ModelTrainer(converted_data_dict,self.model_save_path, self.encoder_save_path, self.decoder_save_path)
            model_trainer.compile_fit_save()
            logging.info("model training complete")
        except Exception as e:
            raise CustomException(e,sys)
    
if __name__ == "__main__":
    train_pipeline = TrainingPipeline()
    train_pipeline.train()