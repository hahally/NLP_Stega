from copy import deepcopy
import random
import torch
import torch.nn.functional as F
from models.model import BinsLstm
from scripts.Constants import EOS, UNK
from scripts.data_loader import create_dataloader, read_file
from scripts.utils import load_model, load_vocab, save_text

class Bins():
    def __init__(self,
                 bins=None,
                 common_tokens=None,
                 bits=None):
        self.bins = bins
        self.common_tokens = common_tokens
        self.bits = bits
        
    def __call__(self, pred_prob):
        """
        input: 
               pred_prob: 为模型预测概率分布, tensor
               bit: 秘密信息,默认为已转为十进制的, int
        output: 
               state: 表示是否嵌入秘密信息, bool
               bit_word_index: 嵌入秘密信息时的token, tensor
        """
        bit = self.bits[0]
        state = True
        targ = []
        for i in range(len(self.bins)):
            if i==bit:
                targ += self.bins[i]
                break
            
        mask = torch.zeros_like(pred_prob, device=pred_prob.device) - 1
        mask[:,targ] = 0
        pred_prob += mask
        bit_word_index = torch.argmax(pred_prob, dim=1)
        if int(bit_word_index[0].item()) in self.common_tokens:
            state = False
        
        if state:
            self.bits.pop(0)
        
        return state, bit_word_index

def get_common_tokens(n):
    from process_data import read_file
    lines = read_file(file='./data/quora/train.txt') + read_file(file='./data/quora/valid.txt')
    word2idx, idx2word = load_vocab('./data/quora/vocab.pkl')
    word_count = dict.fromkeys(word2idx, 0)
    for line in lines:
        for word in line.split():
            if word_count.get(word) is None:
                continue
            word_count[word] += 1
    word_count = sorted(word_count.items(),key=lambda elem: elem[1], reverse=True)
    common_tokens = [word2idx[w_c[0]] for w_c in word_count[:n]]
    
    return common_tokens

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def 统计句子开头单词(files):
    lines = []
    for file in files:
        lines += read_file(file)
    
    first_words = {}
    for line in lines:
        w = line.split()[0]
        if first_words.get(w) is None:
            first_words[w] = 1
        else:
            first_words[w] += 1
    
    first_words = sorted(first_words.items(),key=lambda elem: elem[1], reverse=True)    
    return [w_c[0] for w_c in first_words[:50]]

if __name__ == '__main__':
    setup_seed(2023)
    word2idx, idx2word = load_vocab('./data/quora/vocab.pkl')
    valid_data = create_dataloader(word2idx,'./data/quora/quora-test-src.txt',clip_len=50,batch_size=1,shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = './saved_model/model.pth'
    model = BinsLstm(vocab_size=len(word2idx),
                     embedding_dim=300,
                     hidden_size=256,
                     num_layers=2,
                     dropout=0.3).to(device)
    
    model = load_model(model, checkpoint_path)
    
    first_words = 统计句子开头单词(files=[
        './data/quora/quora-train-src.txt',
        './data/quora/quora-train-tgt.txt',
        './data/quora/quora-val-src.txt',
        './data/quora/quora-val-tgt.txt'])
    
    # bins
    k = 1
    common_tokens = get_common_tokens(10) + [EOS]
    tokens = list(set(list(word2idx.values())) - set(common_tokens))
    random.shuffle(tokens)
    words_in_bin = int(len(tokens) / (2**k))
    bins = [tokens[i:i + words_in_bin]+common_tokens for i in range(0, len(tokens), words_in_bin)]
    bits = [random.choice(list(range(2**k))) for i in range(3000)]
    
    handle = Bins(bins=bins, common_tokens=common_tokens, bits=bits.copy())
    sents = []
    normal_sents = []
    max_len = 30
    while len(handle.bits):
        first_word = random.choice(first_words)
        input_data_1 = torch.LongTensor([[word2idx.get(first_word, UNK)]]).to(device)
        input_data_2 = torch.LongTensor([[word2idx.get(first_word, UNK)]]).to(device)
        
        model.eval()
        with torch.no_grad():
            tgt_normal = torch.ones_like(input_data_2) * input_data_2
            tgt_normal = tgt_normal.long()
            
            tgt = torch.ones_like(input_data_1) * input_data_1
            tgt = tgt.long()
            
            hidden_1 = model.init_hidden(1)
            hidden_1 = (hidden_1[0].to(device),
                      hidden_1[1].to(device))
            
            hidden_2 = deepcopy(hidden_1)
            for i in range(max_len):
                out, hidden_1 = model(input_data_1, hidden_1)
                out = F.softmax(out, dim=-1)
                pred_prob = out[:,-1]
                if len(handle.bits)==0:
                    break
                state, bit_word_index = handle(pred_prob=pred_prob)
                if int(bit_word_index[0].item()) != EOS:
                    tgt = torch.cat([tgt, bit_word_index.unsqueeze(1)], dim=1)
                    input_data_1 = tgt[:,-1:]
                
                else:
                    break
                
                # 正常解码
                out, hidden_2 = model(input_data_2, hidden_2)
                out = F.softmax(out, dim=-1)
                word_pre = out[:,-1]
                word_index = torch.multinomial(word_pre, num_samples=1, replacement=False).squeeze(1)
                tgt_normal = torch.cat([tgt_normal, word_index.unsqueeze(1)], dim=1)
                input_data_2 = tgt_normal[:,-1:]
                
            # idx2word
            sent = ' '.join([idx2word[int(idx)] for idx in tgt[0]])
            sents.append(sent.replace('<EOS>','').strip())
            sent = ' '.join([idx2word[int(idx)] for idx in tgt_normal[0]])
            normal_sents.append(sent.replace('<EOS>','').strip())
            
    save_text(sents, './outputs-bins.txt')
    save_text(normal_sents, './outputs-multinomial.txt')
    
    
    # 秘密信息提取
    mes = ''
    for s in sents:
        for w in s.split()[1:]:
            if word2idx[w] in common_tokens:
                continue
            for i in range(len(bins)):
                if word2idx[w] in bins[i]:
                    mes += str(i)

    for i,j in zip(mes, bits):
        if i!=str(j):
            print('False',type(i),type(j))
            exit(0)
