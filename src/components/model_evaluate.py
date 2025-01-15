import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
from keras.models import load_model
from src.utils import load_object
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical



class ModelPredict:
    def __init__(self, encoder_model_path, decoder_model_path, input_tokenizer_path, evaluate_values_path):
        try:
            evaluate_values = load_object(evaluate_values_path)
            self.target_token_index = evaluate_values["target_token_index"]
            self.max_decoder_seq_length = evaluate_values["max_decoder_seq_length"]
            self.max_encoder_seq_length = evaluate_values["max_encoder_seq_length"]
            self.encoder_model = load_model(encoder_model_path)
            self.decoder_model = load_model(decoder_model_path)
            self.input_tokenizer = load_object(input_tokenizer_path)
            self.num_decoder_tokens = evaluate_values["num_decoder_tokens"]
            self.num_encoder_tokens =  evaluate_values["num_encoder_tokens"]
            self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())
        except Exception as e:
            raise CustomException(e,sys)
    def decode_sequence(self, input_seq):
        try:
            # Encode the input sequence to get the states
            states_value = self.encoder_model.predict(input_seq)

            # Generate an empty target sequence of length 1
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, self.target_token_index['\t']] = 1.

            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:
                # Predict the next character and update the states
                output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

                # Apply temperature to predictions for sampling
                temperature = .25
                pred = output_tokens[0, -1, :].astype('float64')
                pred = pred ** (1/temperature)
                pred = pred / np.sum(pred) 
                pred[0] = 0
                sampled_token_index = np.argmax(np.random.multinomial(1,pred,1))
                sampled_char = self.reverse_target_char_index[sampled_token_index]
                decoded_sentence += sampled_char

                # Stop condition: hit max length or find stop character
                if (sampled_char == '\n' or len(decoded_sentence) > self.max_decoder_seq_length):
                    stop_condition = True

                # Update the target sequence for the next character
                target_seq = np.zeros((1, 1, self.num_decoder_tokens))
                target_seq[0, 0, sampled_token_index] = 1.

                states_value = [h, c]

            return decoded_sentence
        except Exception as e:
            raise CustomException(e,sys)
    def sentence_onehot(self, input_sentence):
        try:
        # Tokenize the input sentence
            word_index = self.input_tokenizer.word_index
            seq = self.input_tokenizer.texts_to_sequences([input_sentence])
            self.input_tokenizer.word_index = word_index
        # Pad the sequence to match max_len
            added_seq = pad_sequences(seq, maxlen=self.max_encoder_seq_length, padding='post')
            onehot_seq = np.zeros((1,self.max_encoder_seq_length, self.num_encoder_tokens))
            onehot_seq[0] = to_categorical(added_seq, num_classes=self.num_encoder_tokens)
            return onehot_seq
        except Exception as e:
            raise CustomException(e,sys)         
    def predict(self,input_sentence):
        try: 
            sequence = self.sentence_onehot(input_sentence)
            return self.decode_sequence(sequence)
        except Exception as e:
            raise CustomException(e,sys)    
            