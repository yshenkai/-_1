# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 21:10:09 2018

@author: shenkai

TO: keeping extending
"""

from gensim.models import word2vec
import logging
import multiprocessing

logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s",level=logging.INFO)
sentenes=word2vec.LineSentence("wiki.zh.seg.utf.txt")
model=word2vec.Word2Vec(sentenes,size=300,window=5,min_count=2,workers=multiprocessing.cpu_count())

model.wv.save_word2vec_format("test.txt",binary=False)