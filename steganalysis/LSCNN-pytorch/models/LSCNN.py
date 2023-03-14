import torch
import torch.nn as nn

class Lscnn(nn.Module):
    def __init__(self, 
                 init_emb=None, 
                 embedding_dim=300,
                 vocab_size=15300,
                 lr_mode='dynamic',
                 kernel_size=[3,5,7], 
                 kernel_num=128,
                 num_classes=2,
                 drop_rate=0.5):
        super(Lscnn, self).__init__()
        self.lr_mode = lr_mode # dynamic, static, both
        assert lr_mode in ['static','dynamic','both'],"lr_mode..."
        # self.word_emb = None
        # self.w2v = None
        if lr_mode=='dynamic':
            self.word_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        if lr_mode=='static':
            assert init_emb != None,'init_emb...'
            self.w2v = nn.Embedding.from_pretrained(embeddings=init_emb,freeze=True)
        if lr_mode=='both':
            assert init_emb != None,'init_emb'
            self.word_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
            self.w2v = nn.Embedding.from_pretrained(embeddings=init_emb,freeze=True)
            self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
            
        self.conv1 = nn.ModuleList([nn.Conv1d(
            in_channels=embedding_dim, 
            out_channels=kernel_num,
            kernel_size=size) for size in kernel_size])
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Sequential(nn.Linear(in_features=kernel_num*len(kernel_size),out_features=num_classes))
        
    def forward(self, x):
        out = []
        if self.lr_mode=='dynamic':
            x = self.word_emb(x)
        if self.lr_mode=='static':
            x = self.w2v(x)
        if self.lr_mode=='both':
            x1 = self.word_emb(x)
            x2 = self.w2v(x)
            x = torch.stack([x1,x2],dim=1)
            x = self.conv(x).squeeze(dim=1)
            
        for conv in self.conv1:
            out.append(conv(x.transpose(2,1)).max(dim=-1)[0])
        
        out = torch.cat(out,dim=-1)
        out = self.fc(self.dropout(out))
        
        return out
            
        
if __name__=='__main__':
    x = torch.tensor([range(0,32),range(0,32)],dtype=torch.long)
    init_emb = torch.randn((100,256))
    model = Lscnn(init_emb=init_emb,lr_mode='dynamic',vocab_size=100,embedding_dim=256)
    model(x)
