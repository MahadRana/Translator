import re
import string
import os
import sys
from src.exception import CustomException
from src.logger import logging
from unicodedata import normalize
import numpy as np 
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self,path):
        if not os.path.exists(path):
            raise CustomException(f"File not found at {path}", sys)
        self.path = path
    def load_doc(self):
        try:
            with open(self.path, mode='rt', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise CustomException(e,sys)

    # Parse document into pairs of sentences
    def to_pairs(self, doc):
        try:
            lines = doc.strip().split('\n')
            pairs = [line.split('\t') for line in  lines]
            return pairs
        except Exception as e:
            raise CustomException(e,sys)

    # Clean text data by normalizing and removing non-alphabetic characters
    def clean_data(self, lines):
        try:
            cleaned = list()
            re_print = re.compile('[^%s]' % re.escape(string.printable))
            table = str.maketrans('', '', string.punctuation)
            for pair in lines:
                clean_pair = list()
                for line in pair:
                    line = normalize('NFD', line).encode('ascii', 'ignore')
                    line = line.decode('UTF-8')
                    line = line.split()
                    line = [word.lower() for word in line]
                    line = [word.translate(table) for word in line]
                    line = [re_print.sub('', w) for w in line]
                    line = [word for word in line if word.isalpha()]
                    clean_pair.append(' '.join(line))
                cleaned.append(clean_pair)
            return np.array(cleaned)
        except Exception as e:
            raise CustomException(e,sys)
    def preprocess(self):
        try:
            doc = self.load_doc()
            pairs = self.to_pairs(doc)
            data = self.clean_data(pairs)
            return data
        except Exception as e:
            raise CustomException(e,sys)
    
    def split(self, train_size=0.8, test_size=.2):
        try:
            if not np.isclose(train_size + test_size, 1.0):
                raise CustomException("Train, validation, and test sizes must sum to 1.", sys)
            data = self.preprocess()
            train_data, test_data = train_test_split(data, train_size=train_size)
            input_texts_train = train_data[:, 0]
            target_texts_train = ['\t' + text + '\n' for text in train_data[:, 1]]
            input_texts_test = test_data[:, 0]
            target_texts_test = ['\t' + text + '\n' for text in test_data[:, 1]]
            max_encoder_seq_length = max(len(line) for line in input_texts_train + input_texts_test)
            max_decoder_seq_length = max(len(line) for line in target_texts_train + target_texts_test)
            return {
                "train": {"input_texts": input_texts_train, "target_texts": target_texts_train},
                "test": {"input_texts": input_texts_test, "target_texts": target_texts_test},
                "max_lengths": {
                    "encoder": max_encoder_seq_length,
                    "decoder": max_decoder_seq_length
                }
            }
        except Exception as e:
            raise CustomException(e,sys)
    