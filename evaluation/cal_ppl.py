import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import BertForMaskedLM

# 在用transformers中的BertForMaskedLM来预测被mask掉的单词时一定要加特殊字符[ C L S ] 和 [ S E P ] [CLS]和[SEP][CLS]和[SEP]。不然效果很差很差！！！

def cpt_ppl(sentence, model, tokenizer):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        tokenize_input = tokenizer.tokenize(sentence)
        # print('tokenize_input',tokenize_input)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        # print('tensor_input',tensor_input)
        sen_len = len(tokenize_input)
        sentence_loss = 0.

        mask_ids = tokenizer.convert_tokens_to_ids('[MASK]')
        d1 = torch.diag_embed(torch.ones(1,sen_len)*mask_ids)
        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        input_ids = torch.ones((sen_len,sen_len))*input_ids
        diag_ids = torch.diag(input_ids)
        a_diag = torch.diag_embed(diag_ids)
        mask_input = diag_ids -  a_diag + d1
        mask_input = mask_input[0,:,:]
        mask_input = mask_input.long().to(DEVICE)
        cls_token = torch.ones(mask_input.shape[0],1) * 101
        cls_token = cls_token.long().to(DEVICE)
        sep_token = torch.ones(mask_input.shape[0],1) * 102
        sep_token = cls_token.long().to(DEVICE)
        mask_input = torch.cat([cls_token,mask_input,sep_token],dim=1)
        outputs = model(mask_input)
        
        idx = torch.arange(0,sen_len)
        idx = idx.long()
        pre_scores = outputs[0][:,1:-1,:][idx,idx]
        softmax = nn.Softmax(dim=1)
        ps_norm = softmax(pre_scores).cpu().log()
        sentence_loss = ps_norm[idx,tensor_input[0]].sum()
        ppl = torch.exp(-sentence_loss / sen_len)
        ppl = round(ppl.item(), 4)
        return ppl

if __name__ == '__main__':
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    filepath = r"../result/quora/quora-test-src.txt" # 计算的文本
    ppl_result_path = r"./原始文本_ppl_CP.txt"

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    count = 0
    with open(ppl_result_path, 'w+', encoding='utf-8') as f:
        for line in tqdm(lines):
            ppl = cpt_ppl(line.strip(), model, tokenizer)
            f.write(line[:-1] + ' ' + str(ppl) + '\n')
            count += ppl

        f.write("avg ppl:" + str(round(count / len(lines),2)))
        print("avg ppl:", round(count / len(lines),2))

