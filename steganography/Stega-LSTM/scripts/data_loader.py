import torch
from torch.utils.data import Dataset, DataLoader
from scripts.Constants import *
class BinsDataset(Dataset):
    def __init__(self,
                 word2idx,
                 file,
                 clip_len = 50):
        super(BinsDataset).__init__()
        
        self.data = read_file(file)
        self.word2idx = word2idx
        self.clip_len = clip_len
        self.UNK = UNK
        self.PAD = PAD
        self.BOS = BOS
        self.EOS = EOS
    
    def __getitem__(self, index):
        sent = self.data[index].split()
        sent.append(EOS_WORD)
        # sent.insert(0,BOS_WORD)
        sent_token = [self.word2idx.get(word, UNK) for word in sent]
        
        return sent_token
    
    def __len__(self):
        
        return len(self.data)

def read_file(file):
    lines = []
    with open(file=file, mode='r', encoding='utf-8') as f:
        line = f.readline().strip()
        while line:
            lines.append(line)
            line = f.readline().strip()
            
    return lines

def padding(data, max_len, padding_value=0):
    data = data + [padding_value] * (max_len-len(data))
    
    return data
    
def collate_fn(batch):
    max_len = max([len(one) for one in batch])
    batch = [padding(dt, max_len, padding_value=0) for dt in batch]
    
    return torch.LongTensor(batch)

def create_dataloader(tokenizer,
                      file,
                      clip_len=20,
                      batch_size=1,
                      shuffle=True,
                      num_workers=0,
                      pin_memory=False):

    dataset = BinsDataset(tokenizer, file, clip_len)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=False,
                            collate_fn=collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    
    return dataloader