import numpy as np

import keras
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, CuDNNLSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class LSTMEmbedding:
    def __init__(self) -> None:
        self.vocab_len = 0
        self.embed_vector_len = 0
        self.maxLen = 0
        self.tokenizer = None
        self.word_to_vec_map = None 


    def initialize_tokenizer(self, num_words, maxLen) -> None:
        self.tokenizer = Tokenizer(num_words=num_words)
        self.maxLen = maxLen
        
    def fit_tokenizer(self, X_train) -> None:
        self.tokenizer.fit_on_texts(X_train)

        self.vocab_len = len(self.tokenizer.word_index)

    def read_glove_vector(self, glove_vec) -> None:
        with open(glove_vec, 'r', encoding='UTF-8') as f:
            #words = set()
            word_to_vec_map = {}
            for line in f:
                w_line = line.split()
                curr_word = w_line[0]
                word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)

        self.embed_vector_len = word_to_vec_map['twitter'].shape[0]
        
        self.word_to_vec_map = word_to_vec_map

    def text_to_sequences(self, input_text) -> np.ndarray:
        X_indices = self.tokenizer.texts_to_sequences(input_text)
        X_indices = pad_sequences(X_indices, maxlen=self.maxLen, padding='post')

        return X_indices


    def create_embedding_layer(self, trainable=False) -> keras.layers.Layer: 
        emb_matrix = np.zeros((self.vocab_len, self.embed_vector_len))
        words_to_index = self.tokenizer.word_index


        for word, index in words_to_index.items():
            embedding_vector = self.word_to_vec_map.get(word)
            if embedding_vector is not None:
                emb_matrix[index, :] = embedding_vector
        embedding_layer = Embedding(input_dim=self.vocab_len, output_dim=self.embed_vector_len, 
                                    input_length=self.maxLen, weights = [emb_matrix], trainable=trainable)

        return embedding_layer

    
    def build_model(self, hidden_nodes, input_shape, embedding_layer,
                     bidirectional=False, dropout_rate = 0.2, seed = 42) -> Sequential:
    
        tf.random.set_seed(seed)
        
        model = Sequential()
        
        model.add(embedding_layer)
        
        return_sequences = len(hidden_nodes)>1
        model.add(self.lstm_layer(hidden_nodes[0], return_sequences, bidirectional))
        #dropout 0.2
        if return_sequences:
            for n_nodes in hidden_nodes[1:-1]:
                model.add(Dropout(dropout_rate))
                model.add(self.lstm_layer(n_nodes, True, bidirectional))
                
            model.add(Dropout(dropout_rate))
            model.add(self.lstm_layer(hidden_nodes[-1], False, bidirectional))
                
        model.add(Dense(1, activation='sigmoid'))
        
        return model


    def lstm_layer(self, n_nodes, return_sequence, bidirectional) -> keras.layers.Layer:
            try:
                if len(tf.config.list_physical_devices('GPU')) > 0:
                    lstm = CuDNNLSTM
                else:
                    lstm = LSTM
            except:
                lstm = LSTM

            if bidirectional:
                return Bidirectional(CuDNNLSTM(n_nodes, return_sequences=return_sequence))
            else:
                return CuDNNLSTM(n_nodes, return_sequences=return_sequence)