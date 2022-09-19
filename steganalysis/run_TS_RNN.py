import tensorflow as tf
import numpy as np
import random
from TS_RNN.train import train_model
from TS_RNN.inference import infer
import pandas as pd

random.seed(2022)
np.random.seed(2022)
tf.random.set_seed(2022)




if __name__ == '__main__':

    result = './result.txt'
    f = open(file=result, mode='w', encoding='utf-8')
    mode = 'rnn' # 或 birnn
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
    
    # train_model(train_data,
    #             tokens,
    #             mode=mode,
    #             k_folds=k_folds,
    #             model_path='./TS_RNN/model_dir',
    #             lr_rate=0.001,
    #             batch_size=128,
    #             expoch=200,
    #             max_len=32,
    #             num_classes=2,
    #             random_state=2022)
    
    # 推理
    pre = infer(test_data,
                                            tokens,
                                            val_mode='inference',
                                            folds=k_folds,
                                            mode=mode,
                                            model_path='./TS_RNN/model_dir',
                                            max_len=32,
                                            num_classes=2)
    
    # 测试评估
    accuracy, f1_s, precision, recall, cm = infer(test_data,
                                            tokens,
                                            val_mode='test',
                                            folds=k_folds,
                                            mode=mode,
                                            model_path='./TS_RNN/model_dir',
                                            max_len=32,
                                            num_classes=2)
    f.write(f'accuracy:{accuracy}, f1_s:{f1_s}, precision:{precision}, recall:{recall}')
