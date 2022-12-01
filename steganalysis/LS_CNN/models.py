import tensorflow as tf
import numpy as np
import random

random.seed(2022)
tf.random.set_seed(2022)
np.random.seed(2022)

class LScnn(tf.keras.Model):
    def __init__(self,
               w2v_emb,
               num_classes=2,
               max_len=32,
               lr_mode='dynamic',
               vocab_size=15300,
               embed_size=300,
               kernel_sizes=[3, 5, 7],
               kernel_num=128,
               drop_rate=0.5
               ):
        super(LScnn, self).__init__()
        assert lr_mode in ['static', 'dynamic', 'both']
        self.max_len = max_len
        self.lr_mode = lr_mode
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.kernel_sizes = kernel_sizes
        self.kernel_num = kernel_num

        self.w2v_emb = tf.keras.layers.Embedding(self.vocab_size, self.embed_size, weights=[w2v_emb], trainable=False)
        
        self.init_conv = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=2021)
        self.init_emb = tf.random_uniform_initializer(minval=-1, maxval=1, seed=2021)
        self.init_dense = tf.initializers.GlorotUniform()

        self.embed = tf.keras.layers.Embedding(self.vocab_size, embed_size, embeddings_initializer=self.init_emb, trainable=True)
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)
        self.conv1_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, kernel_initializer=self.init_conv)
        
        self.conv2 = []
        self.max_pool = []
        for k in self.kernel_sizes:
            self.conv2.append(tf.keras.layers.Conv1D(filters=self.kernel_num,kernel_size=k, strides=1, kernel_initializer=self.init_conv))
            self.max_pool.append(tf.keras.layers.GlobalMaxPooling1D())
            
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax',kernel_initializer=self.init_dense)

    def call(self, x):
        bs = tf.shape(x)[0]
        # print(x.shape[0])
        A = tf.fill([bs,self.max_len,self.embed_size],0.)
        B = tf.fill([bs,self.max_len,self.embed_size],0.)

        if self.lr_mode == 'dynamic':
            A = self.embed(x)
        if self.lr_mode == 'static':
            B = self.w2v_emb(x)
#             B = tf.nn.embedding_lookup(self.w2v_emb, x)

        if self.lr_mode == 'both':
            A = self.embed(x)
            B = self.w2v_emb(x)
#             B = tf.nn.embedding_lookup(self.w2v_emb, x)
        B = tf.cast(B,A.dtype)
        emb = tf.stack([A, B], axis=-1)
        H = self.conv1_1(emb)
        H = tf.reshape(H,shape=tf.shape(H)[:-1])
        pool_output = []
        for conv, pool in zip(self.conv2, self.max_pool):
            c = conv(H)
            p = pool(c)
            pool_output.append(p)
        pool_output = tf.concat([p for p in pool_output], axis=1)
        x_flatten = self.flatten(pool_output)
        out = self.dropout(x_flatten)
        out = self.dense(x_flatten)
        
        return out
