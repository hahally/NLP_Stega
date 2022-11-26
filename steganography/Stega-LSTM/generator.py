import torch
from tqdm import tqdm
from models.model import BinsLstm
from scripts.Constants import *
import torch.nn.functional as F
from scripts.data_loader import create_dataloader
from scripts.utils import load_model, load_vocab, save_text

class Generator():
    def __init__(self, idx2word, model, device, max_len=50, Stega_type=None):
        super(Generator).__init__()
        self.id2word = idx2word
        self.model = model
        self.max_len = max_len
        self.device = device
    
    def generate_stega(self, dataloader, Stega_type):
        sents = []
        if not Stega_type:
            sents = self.generate(dataloader)
        if Stega_type == 'Bins':
            sents = ''
        
        return sents
    
    def generate(self, dataloader):
        sents = []
        self.model.eval()
        with torch.no_grad():
            for input_data in tqdm(dataloader):
                input_data = input_data.to(self.device)
                input_data = input_data[:,:1]
                tgt_tokens, EOS_index = self.generate_batch(input_data)
                sents += self.idx2sent(tgt_tokens, EOS_index)
        
        return sents
        
    def generate_batch(self, input_data):
        bsz = input_data.size(0)
        # tgt = input_data
        EOS_flag = torch.BoolTensor(bsz).fill_(False).to(input_data.device)
        EOS_index = torch.LongTensor(bsz).fill_(self.max_len).to(input_data.device)
        flag = False
        hidden = self.model.init_hidden(bsz)
        hidden = (hidden[0].to(input_data.device), 
                hidden[1].to(input_data.device))
        for i in range(self.max_len):
            out, hidden = self.model(input_data, hidden)
            out = F.softmax(out, dim=-1)
            word_pre = out[:,-1]
            word_index = torch.argmax(word_pre, dim=1)
            input_data = torch.cat([input_data, word_index.unsqueeze(1)], dim=1)
            mask = (word_index == EOS).view(-1).masked_fill_(EOS_flag, False)
            EOS_flag |= mask
            EOS_index.masked_fill_(mask, i + 1)
            if EOS_flag.sum() == bsz and flag:
                break
            flag = True
        return input_data, EOS_index
    
    def idx2sent(self, sent_tokens, EOS_index):
        sents = []
        for i, tokens in enumerate(sent_tokens):
            tokens = tokens[1:int(EOS_index[i].item())]
            sent = ' '.join([self.id2word[int(idx)] for idx in tokens])
            sents.append(sent)
        
        return sents

if __name__ == '__main__':
    word2idx, idx2word = load_vocab('./data/penn/vocab.pkl')
    valid_data = create_dataloader(word2idx,'./data/penn/test.txt',clip_len=50,batch_size=3*10,shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = './saved_model/model.pth'
    model = BinsLstm(vocab_size=len(word2idx),
                     embedding_dim=200,
                     hidden_size=200,
                     num_layers=2,
                     dropout=0.3).to(device)
    
    model = load_model(model, checkpoint_path)
    G = Generator(idx2word=idx2word, model=model, device=device, max_len=30)
    sents = G.generate(dataloader=valid_data)
    print(sents)
