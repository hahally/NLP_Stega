import math

############################  Huffman  #####################################
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

############################  FLC  #####################################
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
    
############################   Bins   #####################################
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

############################   ADG   #####################################
def  ADG(voc_p, u=None):
    sorted_dict= sorted(voc_p, key=lambda x: x[1], reverse = True) # [(w1,p1),...]
    v = [x[0] for x in sorted_dict]
    p = [x[1] for x in sorted_dict]
    
    p_max = p[0]
    first_token = v[0]
    if not u:
        u = 2**(math.floor(-math.log2(p_max)))
        k = math.floor(-math.log2(p_max))
    mean = 1./u
    group = {}
    for i in range(1,u):
        first = sorted_dict.pop(0)
        first_token = first[0]
        first_p = first[1]
        G = [first_token]
        p_g = [first_p]
        while sum(p_g) < mean:
            e = mean - sum(p_g)
            rest_p = [x[1] for x in sorted_dict]
            abs_dis = list(map(lambda x: abs(x-e), rest_p))
            idx = abs_dis.index(max(abs_dis))
            nearest = sorted_dict[idx]
            nearest_token = nearest[0]
            nearest_p = nearest[1]
            if nearest_p - e < e:
                G.append(nearest_token)
                sorted_dict.pop(idx)
                p_g.append(nearest_p)
            else:
                break
            
        rest_p = [x[1] for x in sorted_dict]
        mean = sum(rest_p)/(u-i)
        
        group[f'group_{i-1}'] = G
        
    v = [x[0] for x in sorted_dict]
    group[f'group_{u-1}'] = v
    return group, u, k

# 获取同义词
from nltk.corpus import wordnet
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lm in syn.lemmas():
            syn_word = lm.name()
            syn_word = syn_word.replace('_', ' ')
            syn_word = syn_word.lower()
            if syn_word==word:continue
            if syn_word in synonyms:continue
            synonyms.append(syn_word)
    
    return list(set(synonyms))

############################   SaBins   ######################################
import random
def SaBins(voc_prob, L=3, eos_token='<eos>'):
    # 假设已经在语料库上进行了 词频统计 voc_prob 并且降序排列了
    # voc_prob: [(w1,p1),...,]
    voc = [vp[0] for vp in voc_prob]
    
    # 初始化分组 Initialize
    num_groups = 2**L
    bins = {str(i):set() for i in range(num_groups)}
    bins[str(2**L)] = {eos_token}
    if eos_token in voc:
        voc.remove(eos_token)
    # Mark all tokens in V as unprocessed
    
    for i in range(len(voc_prob)):
        # Determine Cpi
        C_pi = []
        V_pi = voc_prob[i][0]
        C_pi.append(V_pi)
        C_pi += get_synonyms(word=V_pi)
        # Collect all unprocessed tokens in Cpi
        tmp_C_pi = list(filter(lambda x: x in voc, C_pi))
        while len(tmp_C_pi)>0:
            ns = min(len(tmp_C_pi), 2**L)
            # Randomly select ns tokens
            select_ns_tokens = random.sample(tmp_C_pi, k=ns)
            # Randomly select ns subsets
            select_ns_subsets = random.sample(range(num_groups), k=ns)
            # Randomly assign the ns tokens to the ns subsets
            for token, sub in zip(select_ns_tokens, select_ns_subsets):
                bins[str(sub)].add(token)
                voc.remove(token)
                tmp_C_pi.remove(token)
                
    return bins
