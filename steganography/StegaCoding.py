import math

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
# SaBins
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
