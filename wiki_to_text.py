import os
import sys
import logging

from gensim import corpora

if __name__=="__main__":
    program=os.path.basename(sys.argv[0])#获取程序名称
    logger=logging.getLogger(program)
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s"%"".join(sys.argv))
    
    output="zhi_wiki_la.txt"
    f=open(output,"w")
    wiki=corpora.WikiCorpus("zhwiki-20180901-pages-articles-multistream.xml.bz2",lemmatize=False,dictionary={})
    i=0
    for text in wiki.get_texts():
        f.write(b"".join(text).decode("utf-8")+"\n")
        i+=1
        if(i%1000==0):
            logger.info("Saved"+str(i)+"articles")
    f.close()
    logger.info("Finished saved"+str(i)+"articles")
    