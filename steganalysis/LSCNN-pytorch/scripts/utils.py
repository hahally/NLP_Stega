from nltk.tokenize import word_tokenize
from collections import Counter

from tqdm import tqdm

def get_tokenize(sent):
    
    return word_tokenize(sent)

def create_vocab(corpus):
    word_list = []
    for s in tqdm(corpus):
        word_list += get_tokenize(s)
    word_count = Counter(word_list)
    word_count = sorted(word_count.items(),key=lambda x:x[1],reverse=True)
    word2idx = {'<PAD>':0,'<UNK>':1}
    for k,v in word_count:
        word2idx[k] = len(word2idx)
    
    return word2idx

def save_txt(lines, f):
    with open(file=f, mode='w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")
    
def read_txt(f):
    lines = []
    with open(file=f, mode='r', encoding='utf-8') as f:
        line = f.readline().strip()
        while line:
            lines.append(line)
            line = f.readline().strip()
            
    return lines

