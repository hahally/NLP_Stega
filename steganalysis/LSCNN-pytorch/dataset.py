import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LscnnDataset(Dataset):
    def __init__(self, df, word2idx, max_len=32, pad=0, unk=1, train_mode=True):
        super(LscnnDataset, self).__init__()
        self.train_mode = train_mode
        self.sents = df.sents.tolist()
        self.labels = df.labels.tolist()
        self.word2idx = word2idx
        self.max_len = max_len
        self.pad= pad
        self.unk=unk
    
    def pad2maxlen(self,x):
            x = x[:self.max_len]
            x = x + [self.pad]*(self.max_len - len(x))
            
            return x
    
    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, idx):
        sent = self.pad2maxlen([self.word2idx.get(word, self.unk) for word in self.sents[idx].split()])
        label = self.labels[idx]
        sent = torch.LongTensor(sent)
        # label = torch.LongTensor([label])
        return sent, label
