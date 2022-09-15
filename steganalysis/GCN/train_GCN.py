import sys
import scripts as utils
from models import model
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


if __name__=='__main__':
    import pandas as pd
    datasets = ['R8', 'R52', 'ohsumed_single_23', 'mr','20ng']
    experiment_dataset = datasets[0]

    print("Experiment on :", experiment_dataset)  # , sys.argv)

    ### HYPER PARAMETERS
    MIN_WORD_COUNT = 10  # the word with frequency less than this num will be remove and consider as unknown word later on (probabily is the k mention in paper)
    NEIGHBOR_DISTANCE = 2  # same as paper

    WORD_EMBED_DIM = 200  # dimension for word embedding
    PRETRAIN_EMBEDDING = True  # use pretrain embedding or not
    PRETRAIN_EMBEDDING_FIX = True# False  # skip the training for pretrain embedding or not
    MODEL_MAX_SEQ_LEN = 0  # the length of text should the model encode/learning, set 0 to consider all

    N_EPOCHS = 200
    WARMUP_EPOCHS = 5  # Try warm up

    ## Training params from paper
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.000001
    EARLY_STOP_EPOCHS = 5  # after n_epochs not improve then stop training
    EARLY_STOP_MONITOR = "loss"  # monitor early stop on validation's loss or accuracy.
    
    f = open('GCN-隐写分析结果.txt', mode='w',encoding='utf-8')
    
    data_pd = utils.load_data() # 参考该文件里数据加载格式 编写适应自己的数据加载代码
    """
    data_pd 为 DataFrame格式
    header : [text, label, target]
    text: 为句子
    label: 为标签
    target: train or test, 指明该行数据是训练集还是测试集
    """
    
    data_pd = utils.data_preprocessing(data_pd)
    data_pd, word2idx = utils.features_extracting(data_pd, minimum_word_count=MIN_WORD_COUNT,
                                            neighbor_distance=NEIGHBOR_DISTANCE)
    NUM_CLASSES = len(set(data_pd['y'].values))
    text_level_gnn = model.TextLevelGNN_Model(word2idx, NUM_CLASSES, WORD_EMBED_DIM, PRETRAIN_EMBEDDING,
                                        PRETRAIN_EMBEDDING_FIX, MODEL_MAX_SEQ_LEN)
    text_level_gnn.set_dataset(data_pd)

    text_level_gnn.train_eval(N_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EARLY_STOP_EPOCHS,EARLY_STOP_MONITOR, WARMUP_EPOCHS)
    
    preds = text_level_gnn.infer_func(text_level_gnn.data_test,BATCH_SIZE,torch.nn.CrossEntropyLoss().to(text_level_gnn.device))
    
    test = data_pd[data_pd.target == 'test'][['label', 'text','y']].reset_index(drop=True)
    test['pred'] = preds
    test.to_csv('./results.csv',index=False)
    n = len(test)
    acc = round(sum(test.pred==test.y)/n*100, 4)
    f1 = round(f1_score(test.y, test.pred)*100, 4)
    precision = round(precision_score(test.y, test.pred)*100,4)
    recall = round(recall_score(test.y, test.pred)*100, 4)
    print(acc,f1,precision,recall)
