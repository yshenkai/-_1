import os
import jieba
import re
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import  Model,Sequential
from keras.layers import Masking#加载屏蔽层
from keras.layers import Input,Dense,Flatten,Activation
from keras.layers import Embedding,LSTM,Bidirectional,Merge,Conv1D,GlobalMaxPooling1D,Reshape,Dropout
from keras.initializers import Constant
from keras.layers import Convolution2D,MaxPooling2D
from keras.utils import to_categorical

def load_glove_model(filename):
    embedding_matrix={}
    with open(filename,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            values=line.split()
            word=values[0]
            embed=np.asarray(values[1:],dtype=np.float32)
            embedding_matrix[word]=embed
    return embedding_matrix

def clean_data(string):
    #使用正则式，rez中的sub替换功能，对数据进行清洗
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    string = re.sub(r"\r\n", "", string)
    string = re.sub(r"\r", "", string)
    string = re.sub(r"\,","",string)
    string = re.sub(r"\.","",string)
    string = re.sub(r"\，","",string)
    string = re.sub(r"\。","",string)
    string = re.sub(r"\（","",string)
    string = re.sub(r"\）","",string)
    string = re.sub(r"\(","",string)
    string = re.sub(r"\)","",string)
    string = re.sub(r"\“","",string)
    string = re.sub(r"\”","",string)
    string=re.sub(r"\:","",string)
    string=re.sub(r"\!","",string)
    string=re.sub(r"\?","",string)
    string=string.strip()
    return string

def load_data(filepath):
    texts=[]
    labels=[]
    label_to_index={}
    for name in os.listdir(filepath):
        path=os.path.join(filepath,name)
        label_index=len(label_to_index)
        label_to_index[name]=label_index
        if os.path.isdir(path):
            for fname in os.listdir(path):
                fpath=os.path.join(path,fname)
                with open(fpath,"r",errors="ignore") as f:
                    data=clean_data(f.read())
                    #text=clean_data(text)
                    data=jieba.cut(data)
                text=" ".join(data)
                texts.append(text)    
                labels.append(label_index)
    return texts,labels,label_to_index

model=load_glove_model("test.txt")
texts,labels,labe_to_index=load_data("ChnSentiCorp_htl_unba_10000")
print("len_texts",len(texts))
print("text")
tokenizer=Tokenizer(num_words=2000)
tokenizer.fit_on_texts(texts)
sequence=tokenizer.texts_to_sequences(texts)
pad_seq=pad_sequences(sequence,maxlen=120)
word_index=tokenizer.word_index
print("word_index",word_index)
data=pad_seq
labels=to_categorical(np.asarray(labels))
print("data_shape",data.shape)
print("labels_shape",labels.shape)
word_len=data.shape[0]
print("word_len",word_len)
index=np.arange(word_len)
np.random.shuffle(index)
data=data[index]
labels=labels[index]

val_num=int(word_len*0.2)

trainX=data[:-val_num]
trainY=labels[:-val_num]
print("trainX_shape",trainX.shape)
print("trainY_shape",trainY.shape)

valX=data[-val_num:]
valY=labels[-val_num:]

embedding_matrix=np.zeros((len(word_index)+1,300))
for zh,i in word_index.items():
    embed=model.get(zh)
    if embed is not None:
        embedding_matrix[i]=embed





print(embedding_matrix.shape)



print(val_num)




embed_layer=Embedding(input_dim=len(word_index)+1,output_dim=300,embeddings_initializer=Constant(embedding_matrix),input_length=120,trainable=False)
model=Sequential()
model.add(embed_layer)
model.add(LSTM(120,input_dim=300,input_length=120,return_sequences=True))
model.add(Bidirectional(LSTM(60,return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(30,return_sequences=False)))
model.add(Dropout(0.3))
model.add(Dense(trainY.shape[1],activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.summary()
model.fit(trainX,trainY,validation_data=(valX,valY),epochs=50,batch_size=200)



model.save("qing_gan_fen_xi.model")

#dictionary=gensim.corpora.Dictionary(texts,prune_at=20000000)






                    
    
    

    
            