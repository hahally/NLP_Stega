import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random

random.seed(2022)
tf.random.set_seed(2022)
np.random.seed(2022)

class  Tsrnn(tf.keras.Model):
    def __init__(self,
            max_len=32,
            vocab_size=12000,
            embed_size=256,
            drop_rate=0.5,
            mode = 'RNN',
            num_classes = 3
            ):
        super(Tsrnn, self).__init__()
        assert mode.upper() in ['RNN','BIRNN']
        
        if mode.upper() == 'RNN':
            self.h_num = 3
            self.units = 200
        
        if mode.upper() == 'BIRNN':
            self.h_num = 2
            self.units = 100
        
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.init_emb = tf.random_uniform_initializer(minval=-1, maxval=1, seed=2022)
        self.init_dense = tf.initializers.GlorotUniform()

        self.embed = layers.Embedding(self.vocab_size, self.embed_size, embeddings_initializer=self.init_emb, trainable=True)
        self.lstm_layes = []
        return_sequences = True
        for i in range(self.h_num):
            if i == (self.h_num-1):
                return_sequences = False
            if mode.upper() == 'BIRNN':
                self.lstm_layes.append(layers.Bidirectional(tf.keras.layers.LSTM(units = self.units, return_sequences=return_sequences), merge_mode='concat'))
            if mode.upper() == 'RNN':
                self.lstm_layes.append(layers.LSTM(units = self.units, return_sequences=return_sequences))
            
        self.dropout = layers.Dropout(rate=drop_rate)
        self.dense = layers.Dense(128,kernel_initializer=self.init_dense)
        
        self.out_dense = layers.Dense(num_classes, activation='softmax',kernel_initializer=self.init_dense, kernel_regularizer=tf.keras.regularizers.l2(0.01))
    
    def call(self, x):
        
        out = self.embed(x)
        lstm_out = []
        for lstm in self.lstm_layes:
            out = lstm(out)
            lstm_out.append(out)
        
        out = self.dense(out)
        out = self.dropout(out)
        out = self.out_dense(out)
        
        return out