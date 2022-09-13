import random

# 读取文件
def read_file(file):
    lines = []
    with open(file, mode='r', encoding='utf-8') as f:
        line = f.readline().strip()
        while line:
            if line.isspace():
                continue
            lines.append(line)
            line = f.readline().strip()

    return lines

# 保存文件
def save_text(lines,file):
    with open(file=file, mode='w', encoding='utf-8') as f:
        for line in lines:
            f.write(line+'\n')

# 获取同义词

# def get_synonyms(word):
#     synonyms = []
#     for syn in wordnet.synsets(word):
#         for lm in syn.lemmas():
#             syn_word = lm.name()
#             syn_word = syn_word.replace('_', ' ')
#             syn_word = syn_word.lower()
#             if syn_word==word:continue
#             if syn_word in synonyms:continue
#             synonyms.append(syn_word)
    
    # return list(set(synonyms))

# 获取句子
def get_random_sents(file=None,num=1):
    sents = read_file(file)
    assert num<=len(sents)
    random_sent = random.sample(sents, k=num)
       
    return random_sent
    
# 获取比特流
def get_random_steg_infos(num=1):
    s = ['0','1']
    steg_infos = ''
    for c in range(num):
        steg_infos += random.choice(s)
        
    return steg_infos

def get_sents(lines, index):
    
    return [ lines[idx] for idx in index]

def get_random_index(max_index=20000, num=500):
    index_list = list(range(max_index))
    return random.sample(index_list, k=num)

if __name__ == '__main__':
    seed = 2022
    random.seed(seed)
    
    



    
