import sys 
import numpy as np
from src.exception import CustomException
from src.logger import logging
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import optimizers


class Data_Transformation: 
    def __init__(self, train_data=None, val_data=None, test_data=None):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.input_token_index = None
        self.target_token_index = None
        self.num_encoder_tokens = 0
        self.num_decoder_tokens = 0
    def text2sequences_train(self, max_len):
        try:
            tokenizer = Tokenizer(char_level=True, filters='')
            tokenizer.fit_on_texts(self.train_data)
            seqs = tokenizer.texts_to_sequences(self.train_data)
            seqs_pad = pad_sequences(seqs, maxlen=max_len, padding='post')
            return seqs_pad, tokenizer
        except Exception as e:
            raise CustomException(e,sys)
        
    def text2sequences_else(self, max_len, tokenizer, data):
        try:
            seqs = tokenizer.texts_to_sequences(data)
            encoder_input_seq_val = pad_sequences(seqs, maxlen=max_len, padding='post')
            tokenizer.word_index = tokenizer.word_index
            return encoder_input_seq_val
        except Exception as e:
            raise CustomException(e,sys)   

    def onehot_encode(self, sequences, max_len, vocab_size):
        try:
            n = len(sequences)
            data = np.zeros((n, max_len, vocab_size))
            for i in range(n):
                data[i, :, :] = to_categorical(sequences[i], num_classes=vocab_size)
            return data
        except Exception as e:
            raise CustomException(e,sys)
    def getTargetData(self, data, max_len, num_tokens):
        try:
            decoder_target_seq = np.zeros(data.shape)
            decoder_target_seq[:, 0:-1] = data[:, 1:]
            decoder_target_data = self.onehot_encode(decoder_target_seq, max_len, num_tokens)
            return decoder_target_data
        except Exception as e:
            raise CustomException(e,sys)
    
    def transform_train(self):
        try:
            input_texts = self.data[:, 0]
            target_texts = ['\t' + text + '\n' for text in self.data[:, 1]]
            max_encoder_seq_length = max(len(line) for line in input_texts)
            max_decoder_seq_length = max(len(line) for line in target_texts)
            encoder_input_seq, input_token_index = self.text2sequences(max_encoder_seq_length, input_texts)
            decoder_input_seq, target_token_index = self.text2sequences(max_decoder_seq_length, target_texts)
            num_encoder_tokens = len(input_token_index) + 1
            num_decoder_tokens = len(target_token_index) + 1
            self.input_token_index = input_token_index
            self.target_token_index = target_token_index
            self.num_encoder_tokens = num_encoder_tokens
            self.num_decoder_tokens = num_decoder_tokens
            encoder_input_data = self.onehot_encode(encoder_input_seq, max_encoder_seq_length, num_encoder_tokens)
            decoder_input_data = self.onehot_encode(decoder_input_seq, max_decoder_seq_length, num_decoder_tokens)
            decoder_target_data = self.getTargetData(decoder_input_data, max_decoder_seq_length, num_decoder_tokens)
            return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index
        except Exception as e:
            raise CustomException(e,sys)
    