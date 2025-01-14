import sys
import os
import numpy as np
from src.exception import CustomException
from src.logger import logging
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
from keras import optimizers

class ModelTrainer:
    def __init__(self, data_dict, model_save_path, encoder_save_path, decoder_save_path):
        """
        Initialize the ModelTrainer with training data and save path.
        """
        self.data_dict = data_dict
        self.model_save_path = model_save_path
        self.encoder_save_path = encoder_save_path
        self.decoder_save_path = decoder_save_path
        self.latent_dim = 256 
    def encoder(self):
        try:
            encoder_inputs = Input(shape=(None, self.data_dict["tokens"]["encoder"]), name='encoder_inputs')
            encoder_bilstm = Bidirectional(LSTM(self.latent_dim, return_state=True, dropout=0.5, name='encoder_lstm'))
            _, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_inputs)

            state_h = Concatenate()([forward_h, backward_h])
            state_c = Concatenate()([forward_c, backward_c])

            encoder_model = Model(inputs=encoder_inputs, outputs=[state_h, state_c], name='encoder')
            return encoder_model
        except Exception as e:
            raise CustomException(e,sys)

    def decoder(self):
        try:
            new_latent_dim = self.latent_dim*2
            decoder_input_h = Input(shape=(new_latent_dim,), name='decoder_input_h')
            decoder_input_c = Input(shape=(new_latent_dim,), name='decoder_input_c')
            decoder_input_x = Input(shape=(None, self.data_dict["tokens"]["decoder"]), name='decoder_input_x')

            decoder_lstm = LSTM(new_latent_dim, return_sequences=True, return_state=True, dropout=0.5, name='decoder_lstm')
            decoder_lstm_outputs, state_h, state_c = decoder_lstm(decoder_input_x, initial_state=[decoder_input_h, decoder_input_c])

            decoder_dense = Dense(self.data_dict["tokens"]["decoder"], activation='softmax', name='decoder_dense')
            decoder_outputs = decoder_dense(decoder_lstm_outputs)

            decoder_model = Model(inputs=[decoder_input_x, decoder_input_h, decoder_input_c], outputs=[decoder_outputs, state_h, state_c], name='decoder')
            return decoder_lstm, decoder_dense, decoder_model
        except Exception as e:
            raise CustomException(e,sys)
    def seq2seq(self):
        try:
            encoder_input_x = Input(shape=(None, self.data_dict["tokens"]["encoder"]), name='encoder_input_x')
            decoder_input_x = Input(shape=(None, self.data_dict["tokens"]["decoder"]), name='decoder_input_x')
            encoder_model = self.encoder()
            decoder_lstm, decoder_dense, decoder_model = self.decoder()
            encoder_final_states = encoder_model([encoder_input_x])
            decoder_lstm_output, _, _ = decoder_lstm(decoder_input_x, initial_state=encoder_final_states)
            decoder_pred = decoder_dense(decoder_lstm_output)    
            model = Model(inputs=[encoder_input_x, decoder_input_x],
                    outputs=decoder_pred,
                    name='model_training')
            return encoder_model,decoder_model,model
        except Exception as e:
            raise CustomException(e,sys)    
    def compile_fit_save(self):
        try:
            encoder_model, decoder_model, model = self.seq2seq()
            model.compile(optimizer=optimizers.Adam(learning_rate=.001), loss='categorical_crossentropy')
            model.fit([self.data_dict["train"]["encoder_input"], self.data_dict["train"]["decoder_input"]], self.data_dict["train"]["decoder_target"], batch_size=64, epochs=50)
            encoder_model.save(self.encoder_save_path)
            decoder_model.save(self.decoder_save_path)
            model.save(self.model_save_path)
            return encoder_model, decoder_model, model
        except Exception as e:
            raise CustomException(e,sys)
