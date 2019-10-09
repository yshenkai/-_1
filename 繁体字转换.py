# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 12:50:46 2018

@author: shenkai

TO: keeping extending
"""

import codecs
import sys
import opencc
fread=codecs.open("zhwiki-articles.txt","r",encoding="utf-8")
cc=opencc.OpenCC("t2s")
i=0

fwrite=codecs.open("zhwiki-segment.txt","w",encoding="utf-8")

def isAlpha(word):
    try:
        return word.encode("ascii").isalpha()
    except UnicodeEncodeError:
        return False

for line in fread:
    text=line.strip()
    i+=1
    print("line"+str(i))
    data=""
    for char in text.split():
        if isAlpha(char):
            continue
        char=cc.convert(char)
        data+=char
    fwrite.write(data+"\n")

fread.close()
fwrite.close()
    
