from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import  pandas as pd
import json

from scipy import spatial
from nltk import word_tokenize

def train_doc2vec():
    documents = []
    with open('./data/IMDb.txt', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            documents.append(line[1].strip())
    with open('./data/AG-data.txt', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            documents.append(line[1].strip())

    tagged_documents = [TaggedDocument(words=word_tokenize(document), tags=[str(i)]) for i, document in
                        tqdm(enumerate(documents))]

    model = Doc2Vec(tagged_documents, vector_size=512, min_count=0, epochs=50)
    pd.to_pickle({"model":model},"doc2vec.pandas_pickle")

train_doc2vec()

stego_file = ''
cover_file = ''
with open(file=stego_file, mode='r', encoding='utf-8') as f:
    test_stego = [word_tokenize(line.strip()) for line in f]

with open(file=cover_file, mode='r', encoding='utf-8') as f:
    test_cover = [word_tokenize(line.strip()) for line in f]

model=pd.read_pickle("doc2vec.pandas_pickle")["model"]

# 使用 doc2vec 计算语义相似度
mean_score = 0
save_data = {0:[], 1:[]}
for stego, cover in tqdm(zip(test_stego, test_cover)):
    stego_vector = model.infer_vector(stego)
    cover_vector = model.infer_vector(cover)
    save_data[0].append(cover_vector)
    save_data[1].append(stego_vector)
    score = 1 - spatial.distance.cosine(stego_vector, cover_vector)
    mean_score += score

mean_score = mean_score/len(test_cover)
print(f"sim score:{mean_score}")

# 保存数据
pd.to_pickle(save_data,"data.pkl")
