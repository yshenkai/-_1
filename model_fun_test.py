import jieba
import gensim
import os
import numpy as np
import sys
import logging
import multiprocessing
from keras.preprocessing.text import Tokenizer
from gensim.models import word2vec
# =============================================================================
# path=os.path.join("ChnSentiCorp_htl_unba_10000\pos")
# texts=[]
# for fname in sorted(os.listdir(path)):
#     fpath=os.path.join(path,fname)
#     if os.path.isfile(fpath):
#         with open(fpath,"r",errors="ignore") as f:
#             text=f.read()
#             texts.append(text)
# 
# 
# 
# cut_texts=[[word for word in jieba.cut(words)] for words in texts]
# 
# dictionary=gensim.corpora.Dictionary(cut_texts,prune_at=200000)#prune_at为控制向量数，也就是说最多对200000个向量进行编码（向量化）
# #dictionary.merge_with(dict2)表示将两个gensim.corpora.Dictionary对象（实体）合并
# 
# 
# #dictionary.token2id={"a":1,"numamn":2}
# dictionary.save_as_text("dict.dict")
# a={"a":1,"numamn":2}
# 
# =============================================================================



test=["我","我喜","我喜欢","我喜欢怀","我喜欢怀大","我喜欢怀大媛","我喜欢怀大媛啊"]
a=[]
for word in test:
    vec=jieba.cut(word)
    data=" ".join(vec)
    a.append(data)

tokenizer=Tokenizer(num_words=2000)
tokenizer.fit_on_texts(a)
sequeue=tokenizer.texts_to_sequences(a)
print(sequeue)



#dictionary=gensim.corpora.Dictionary(words,prune_at=200000)
