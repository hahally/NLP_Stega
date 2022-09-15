from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from bert_score import score

# 计算句子
def eval(reference, candidate, mode='self-bleu'):
    assert mode.lower() in ['self-bleu', 'rouge-1', 'rouge-2', 'rouge-l', 'bert-score', 'meteor']
    sr = None
    if mode=='self-bleu':
        reference = [[reference.split()]]
        candidate = [candidate.split()]
        bleu4 = corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        bleu4 = round(bleu4*100, 2)
        sr =  bleu4

    if 'rouge' in mode:
        rouge = Rouge()
        rs = rouge.get_scores(candidate, reference)[0]
        sr = round(rs[mode]['r']*100, 2)
    
    if mode=='meteor':
        ms = meteor_score([reference], candidate)
        sr = round(ms*100, 2)
    
    if mode=='bert-score':
        # print(candidate, '\n',reference)
        # from bert_score import score
        (P, R, F), hashname = score(candidate, reference, lang="en", return_hash=True, rescale_with_baseline=True)
        sr = F.tolist()
        sr = [ round(s*100,2) for s in sr]
        
    return sr

# 计算文件
class Eval:

    def __init__(self, source_file, reference_file):
        self.source = self.read_file(source_file, reference=False)
        self.source_ref = self.read_file(source_file, reference=True)
        self.reference = self.read_file(reference_file, reference=True)

    def read_file(self, file, reference=False):
        with open(file, 'r', encoding='UTF-8') as f:
            if reference:
                data = [[[word.lower() for word in seq.strip('\n').split()] 
                          for seq in line.strip('\n').split('\t')] for line in f.readlines()]
            else:
                data = [[word.lower() for word in line.strip('\n').split()] for line in f.readlines()]
        return data
    def bleu(self, reference, candidate):
        bleu4 = corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        bleu4 = round(bleu4*100, 2)
        return bleu4

    def meteor(self, reference, candidate):
        reference = [[' '.join(ref[0])] for ref in reference]
        candidate = [' '.join(cand) for cand in candidate]
        ms = 0
        for r, c in zip(reference, candidate):
            ms += meteor_score(r, c)
            
        avg_ms = round(ms/len(candidate)*100, 2)
        return avg_ms
    
    def rouge_score(self, reference, candidate):
        rouge = Rouge()
        reference = [' '.join(ref[0]) for ref in reference]
        candidate = [' '.join(cand) for cand in candidate]
        rs = rouge.get_scores(reference, candidate, avg=True)
        avg_rs_1 = round(rs['rouge-1']['f']*100, 2)
        avg_rs_2 = round(rs['rouge-2']['f']*100, 2)
        avg_rs_L = round(rs['rouge-l']['f']*100, 2)
        
        return avg_rs_1, avg_rs_2, avg_rs_L
    
    def bertscore(self, reference, candidate):
        reference = [' '.join(ref[0][:-1])+ref[0][-1] for ref in reference]
        candidate = [' '.join(cand[:-1])+cand[-1] for cand in candidate]
        model_type='bert-base-uncased'
        (P, R, F), hashname = score(candidate, reference, model_type=model_type, lang="en", return_hash=True, rescale_with_baseline=True)
        
        bs = round(F.mean().item()*100, 2)
        
        return bs
        
    def unc(self, source, candidate):
        cnt = 0
        for i in range(len(candidate)):
            if candidate[i] == source[i]:
                cnt += 1
        return cnt / len(candidate) * 100
    
    def __call__(self, candidate_file, flag=False):
        candidate = self.read_file(candidate_file, reference=False)
        bleu = self.bleu(self.reference, candidate)
        self_bleu = self.bleu(self.source_ref, candidate)
        # unc = self.unc(self.source, candidate)
        meteor = self.meteor(self.reference, candidate)
        avg_rs_1, avg_rs_2, avg_rs_L = self.rouge_score(self.reference, candidate)
        
        bertScore = False
        if flag:
            bertScore = self.bertscore(self.reference, candidate)
        result = {}
        result['self-BLEU'] = self_bleu
        result['BLEU'] = bleu
        result['meteor'] = meteor
        result['rouge-1'] = avg_rs_1
        result['rouge-2'] = avg_rs_2
        result['rouge-L'] = avg_rs_L
        result['bert-score'] = bertScore
        print(result)
        
        return result

if __name__ == '__main__':
    cand_file = f'cand.txt' # 候选文本
    ref_file = f'ref.txt' # 参考文本
    src_file = 'src.txt' # 原始文本
    eval = Eval(src_file,ref_file)
    result = evalcand_file, flag=True)
    print(result)
    
    
