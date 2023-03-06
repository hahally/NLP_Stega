#Huffman Encoding

#Tree-Node Type
from copy import deepcopy
import random
import torch
import torch.nn.functional as F
from models.model import BinsLstm
from scripts.Constants import EOS, UNK
from scripts.data_loader import read_file
from scripts.utils import load_model, load_vocab, save_text

class Node:
    def __init__(self,freq):
        self.left = None
        self.right = None
        self.father = None
        self.freq = freq
    def isLeft(self):
        return self.father.left == self
#create nodes
def createNodes(freqs):
    return [Node(freq) for freq in freqs]

#create Huffman Tree
def createHuffmanTree(nodes):
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key=lambda item:item.freq)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.freq + node_right.freq)
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)
    queue[0].father = None
    return queue[0]
#Huffman encoding
def huffmanEncoding(nodes,root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != root:
            if node_tmp.isLeft():
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father
    return codes

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

class Huffman():
    def __init__(self,
                 k=None,
                 bits=None):
        self.k = k
        self.bits = bits
        
    def extract(self, pred_prob, word_index):
        bit = None
        prob, idx = torch.topk(pred_prob, k=self.k, dim=1)
        word_prob = [[i.item(),j.item()] for i,j in zip(idx[0], prob[0])]
        nodes = createNodes([item[1] for item in word_prob])
        root = createHuffmanTree(nodes)
        codes = huffmanEncoding(nodes, root)
        for i, w_p in enumerate(word_prob):
            if w_p[0] == word_index:
                bit = codes[i]
                break

        return bit
    
    def __call__(self, pred_prob):
        prob, idx = torch.topk(pred_prob, k=self.k, dim=1)
        word_prob = [[i.item(),j.item()] for i,j in zip(idx[0], prob[0])]
        nodes = createNodes([item[1] for item in word_prob])
        root = createHuffmanTree(nodes)
        codes = huffmanEncoding(nodes, root)
        for i,code in enumerate(codes):
            bit = self.bits[:len(code)]
            if bit == code:
                bit_word_index = word_prob[i][0]
                self.bits = self.bits[len(code):]
                break
        if self.extract(pred_prob,bit_word_index)!=bit:
            print("False")
        bit_word_index = torch.LongTensor([bit_word_index]).to(pred_prob.device)
        return bit_word_index

    
    
class FLC():
    def __init__(self,
                 k=None,
                 bits=None):
        self.k = k
        self.bits = bits
    
    def extract(self, pred_prob, word_index):
        bit = None
        prob, idx = torch.topk(pred_prob, k=2**self.k, dim=1)
        word_prob = [[i.item(),j.item()] for i,j in zip(idx[0], prob[0])]
        codes = [str(bin(i))[2:].zfill(self.k) for i in range(2**self.k)]
        for i, w_p in enumerate(word_prob):
            if w_p[0] == word_index:
                bit = codes[i]
                break

        return bit
    
    def __call__(self, pred_prob):
        prob, idx = torch.topk(pred_prob, k=2**self.k, dim=1)
        word_prob = [[i.item(),j.item()] for i,j in zip(idx[0], prob[0])]
        codes = [str(bin(i))[2:].zfill(self.k) for i in range(2**self.k)]
        bit = self.bits[:self.k]
        for i,code in enumerate(codes):
            if bit==code:
                bit_word_index = word_prob[i][0]
                self.bits = self.bits[self.k:]
                break
            
        bit_word_index = torch.LongTensor([bit_word_index]).to(pred_prob.device)
        return bit_word_index
    
if __name__ == '__main__':
    setup_seed(2023)
    word2idx, idx2word = load_vocab('./data/quora/vocab.pkl')
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
    
    sents = []
    normal_sents = []
    max_len = 30
    k = 2 #
    bits = ''.join([str(random.choice(list(range(2)))) for i in range(3000)])
    
    handle = Huffman(k=2**k,bits=bits)
    # handle = FLC(k=k,bits=bits)
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
                
                # 编码方法
                bit_word_index = handle(pred_prob)
                
                if int(bit_word_index[0].item()) != EOS:
                    tgt = torch.cat([tgt, bit_word_index.unsqueeze(1)], dim=1)
                    input_data_1 = tgt[:,-1:]
                
                # 正常解码
                out, hidden_2 = model(input_data_2, hidden_2)
                out = F.softmax(out, dim=-1)
                word_pre = out[:,-1]
                word_index = torch.multinomial(word_pre, num_samples=1, replacement=False).squeeze(1)
                if int(word_index[0].item()) != EOS:
                    tgt_normal = torch.cat([tgt_normal, word_index.unsqueeze(1)], dim=1)
                    input_data_2 = tgt_normal[:,-1:]
                
            # idx2word
            sent = ' '.join([idx2word[int(idx)] for idx in tgt[0]])
            # sents.append(sent.replace('<EOS>','').strip())
            sents.append(sent.strip())
            sent = ' '.join([idx2word[int(idx)] for idx in tgt_normal[0]])
            # normal_sents.append(sent.replace('<EOS>','').strip())
            normal_sents.append(sent.strip())
            
    save_text(sents, './outputs-FLC.txt')
    save_text(normal_sents, './outputs-multinomial.txt')
