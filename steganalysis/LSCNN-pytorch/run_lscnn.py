import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from config.config_lscnn import Config
from models.LSCNN import Lscnn
from scripts.utils import read_txt,save_txt,create_vocab
import pandas as pd
from dataset import LscnnDataset
from torch.utils.data import DataLoader

# 固定种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     np.random.seed(seed)

def evaluate(model, criterion, x,y):
    n = y.shape[0]
    model.eval()
    with torch.no_grad():
        out = model(x)
        # out = out.squeeze(dim=-1)
        loss = criterion(out,y).item()
    prob = torch.softmax(out,dim=-1)
    acc= (prob.argmax(dim=1) == y).sum().item()
    
    return acc/n,loss,prob

# 训练模型
def train_model(config):
    train_cover = read_txt(f=config.cover_file)
    train_stego = read_txt(f=config.stego_file)
    data = pd.DataFrame()
    data['sents'] = train_cover + train_stego
    data['labels'] = [0]*len(train_cover) + [1] * len(train_stego)
    data = data.sample(frac=1).reset_index(drop=True)
    
    train,test = train_test_split(data, test_size=0.2,shuffle=True, stratify=data.labels)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    corpus = sum([read_txt(f) for f in config.corpus],[])
    print(f"create vocab...")
    word2idx = create_vocab(corpus)
    train_data = LscnnDataset(train, word2idx, max_len=32, pad=0, unk=1, train_mode=True)
    test_data = LscnnDataset(test, word2idx, max_len=32, pad=0, unk=1, train_mode=False)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.works,
                              drop_last=False
                              )
    
    valid_loader = DataLoader(dataset=test_data,
                              batch_size=config.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=config.works,
                              drop_last=False
                              )
    # init model
    model = Lscnn(init_emb=None,
                  lr_mode=config.lr_mode,
                  vocab_size=len(word2idx),
                  embedding_dim=config.embedding_dim,
                  kernel_size=config.kernel_size,
                  kernel_num=config.kernel_num,
                  num_classes=config.num_classes,
                  drop_rate=config.drop_rate)
    model = model.to(config.device)
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    c = 0
    best_score = 0
    print(f"steps/epoch:{len(train_loader)}")
    for e in range(1,1+config.epoch):
        if c>config.early_stop: break
        for step, (x,y) in enumerate(train_loader):
            model.train()
            x,y = x.to(config.device),y.to(config.device)
            out = model(x)
            loss = criterion(out,y)
            prob = torch.softmax(out,dim=-1)
            train_acc= (prob.argmax(dim=1) == y).sum().item()/len(y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step==0 or step%config.print_freq==0:
                print(f"Epoch:{e}, step:{step}, train loss:{loss.item()}, train acc:{train_acc}")
    
        # evaluate
        val_acc = 0
        val_loss = 0
        n = len(valid_loader)
        for (x,y) in tqdm(valid_loader):
            x,y = x.to(config.device),y.to(config.device)
            acc,loss,_ = evaluate(model,criterion,x,y)
            val_acc += acc/n
            val_loss += loss/n
        print(f"Epoch:{e}, val loss:{val_loss}, val acc:{val_acc}")
        if best_score<val_acc:
            best_score = val_acc
            c = 0
            torch.save({'model':model.state_dict(),'word2idx':word2idx}, config.save_model_path)
            print(f"saved model {e}")
        else:
            c+=1
        torch.save({'model':model.state_dict(),'word2idx':word2idx}, './saved_model/last_epoch.model')
# 测试模型
def test_model(config):
    pass

# 运行
def main(config):
    if config.train_model:
        train_model(config)
    if config.test_model:
        test_model(config)
        
if __name__=='__main__':
    config = Config()
    setup_seed(config.seed)
    main(config)
    
