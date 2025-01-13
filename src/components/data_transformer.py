import sys 
import numpy as np
from src.exception import CustomException
from src.logger import logging
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class Data_Transformer:
    def __init__(self, data_dict):
        self.data_dict = data_dict
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
    def text2sequences(self, max_len, i_or_t):
        try:
            if i_or_t != "input_texts" and i_or_t != "target_texts":
                raise CustomException("Invalid use of text2sequences",sys)
            tokenizer = Tokenizer(char_level=True, filters='')
            tokenizer.fit_on_texts(self.data_dict["train"][(i_or_t)])
            seqs = tokenizer.texts_to_sequences(self.data_dict["train"][(i_or_t)])
            train_seqs = pad_sequences(seqs, maxlen=max_len, padding='post')
            word_index = tokenizer.word_index

            seqs = tokenizer.texts_to_sequences(self.data_dict["test"][(i_or_t)])
            test_seqs = pad_sequences(seqs, maxlen=max_len, padding='post')
            tokenizer.word_index = word_index

            return train_seqs, test_seqs, word_index
        except Exception as e:
            raise CustomException(e,sys)
    def convert(self):
        try:
            input_train_seqs, input_test_seqs, input_word_index = self.text2sequences(self.data_dict["max_lengths"]["encoder"], "input_texts")
            target_train_seqs, target_test_seqs, target_word_index = self.text2sequences(self.data_dict["max_lengths"]["decoder"], "target_texts")

            num_encoder_tokens_train = len(input_word_index) + 1
            num_decoder_tokens_train = len(target_word_index) + 1

            encoder_input_data_train = self.onehot_encode(input_train_seqs, self.data_dict["max_lengths"]["encoder"], num_encoder_tokens_train)
            decoder_input_data_train = self.onehot_encode(target_train_seqs, self.data_dict["max_lengths"]["decoder"], num_decoder_tokens_train)
            decoder_target_data_train = self.getTargetData(target_train_seqs, self.data_dict["max_lengths"]["decoder"], num_decoder_tokens_train)

            encoder_input_data_test = self.onehot_encode(input_test_seqs, self.data_dict["max_lengths"]["encoder"], num_encoder_tokens_train)
            decoder_input_data_test = self.onehot_encode(target_test_seqs, self.data_dict["max_lengths"]["decoder"], num_decoder_tokens_train)
            decoder_target_data_test = self.getTargetData(target_test_seqs, self.data_dict["max_lengths"]["decoder"], num_decoder_tokens_train)

            return {
                "train": {"encoder_input": encoder_input_data_train, "decoder_input": decoder_input_data_train, "decoder_target":decoder_target_data_train},
                "test": {"encoder_input": encoder_input_data_test, "decoder_input": decoder_input_data_test, "decoder_target":decoder_target_data_test},
                "tokens": {
                    "encoder": num_encoder_tokens_train,
                    "decoder": num_decoder_tokens_train
                }
            }
        except Exception as e:
            raise CustomException(e,sys)