import sys
import os
from src.exception import CustomException
from src.logger import logging
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from statistics import mean


class Model_Evaluate:
    def __init__(self, target_token_index, max_decoder_seq_length, encoder_model, decoder_model, num_decoder_tokens, encoder_input_data_test, target_texts_test):
        self.target_token_index = target_token_index
        self.max_decoder_seq_length = max_decoder_seq_length
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model 
        self.num_decoder_tokens = num_decoder_tokens
        self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())
        self.encoder_input_data_test = encoder_input_data_test
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
    def average_blue(self):
        try:
            bleu_scores = []
            for i in range(100): 
                translation = self.decode_sequence(self.encoder_input_data_test[i:i+1])
                truth = self.target_texts_test[i][1:-1]
                bleu = sentence_bleu([translation], truth)
                print("BLEU: ", bleu)
                bleu_scores.append(bleu)

            # Compute average BLEU score
            avg_bleu = mean(bleu_scores)
            return avg_bleu
        except Exception as e:
            raise CustomException(e,sys)