import sys
from src.exception import CustomException
from src.components.model_evaluate import ModelPredict
class TrainingPipeline:
    def __init__(self):
        self.encoder_model_path = "artifacts/encoder_model.h5"
        self.decoder_model_path = "artifacts/decoder_model.h5"
        self.input_tokenizer_path = "artifacts\input_tokenizer.pkl"
        self.evaluate_values_path = "artifacts\evaluate_values.pkl"
    def predict(self):
        predictModel = ModelPredict(self.encoder_model_path, self.decoder_model_path, self.input_tokenizer_path, self.evaluate_values_path)
        return predictModel.predict()