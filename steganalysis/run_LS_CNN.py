import tensorflow as tf
import numpy as np
import random
from LS_CNN.train import train_model
from LS_CNN.inference import infer
import pandas as pd
import gensim

random.seed(2022)
np.random.seed(2022)
tf.random.set_seed(2022)

if __name__ == '__main__':

    result = './lscnn-result.txt'
    f = open(file=result, mode='w', encoding='utf-8')
    mode = 'both' # 或 birnn
    # 折数
    k_folds = 5
    data = pd.read_csv('./all.csv')
    
    dt1 = data[data.label==0].reset_index(drop=True)
    dt2 = data[data.label==1].reset_index(drop=True)
    
    n1 = int(len(dt1)*0.9)
    n2 = int(len(dt2)*0.9)
    
    train_data = pd.concat([dt1.iloc[:n1],dt2.iloc[:n2]],axis=0).sample(frac=1).reset_index(drop=True)
    test_data = pd.concat([dt1.iloc[n1:],dt2.iloc[n2:]],axis=0).reset_index(drop=True)
    sentences = train_data.sentence.tolist() +test_data.sentence.tolist()
    tokens = tf.keras.preprocessing.text.Tokenizer(num_words=None,
                                                filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                                lower=True,
                                                split=' ',
                                                char_level=False,
                                                oov_token=None,
                                                document_count=0)
    
    tokens.fit_on_texts(sentences)
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./LS_CNN/model_dir/GoogleNews-vectors-negative300.bin',binary=True)
    embedding_matrix = []
    embedding_matrix.append(np.random.uniform(1, -1, 300))
    for w,idx in tokens.word_index.items():
        if w in w2v_model.index_to_key:
            embedding_matrix.append(w2v_model.get_vector(w))
        else:
            embedding_matrix.append(np.random.uniform(1, -1, 300))

    embedding_matrix = np.array(embedding_matrix)
    accuracy, f1_s, precision, recall, cm = train_model(train_data,
                test_data,
                tokens,
                w2v_emb=embedding_matrix,
                mode=mode,
                k_folds=k_folds,
                model_path='./LS_CNN/model_dir',
                lr_rate=0.001,
                batch_size=128,
                expoch=200,
                max_len=32,
                num_classes=2,
                random_state=2022)
    
    # accuracy, f1_s, precision, recall, cm = infer(test_data,
    #                                         embedding_matrix,      
    #                                         tokens,
    #                                         folds=k_folds,
    #                                         mode=mode,
    #                                         model_path='./LS_CNN/model_dir',
    #                                         max_len=32,
    #                                         num_classes=2)
    
    f.write(f'accuracy:{accuracy}, f1_s:{f1_s}, precision:{precision}, recall:{recall}')