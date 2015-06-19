import os,re,thread,time
import glob
import chardet
import profile 

def is_alphabet(rawdata):
    type=chardet.detect(rawdata)
    if type['encoding']=='ascii' and type['confidence']>0.9:
        return True
    else:
        return False

def readFile(path):
    tweet = open(path,"r").read()
    tweet = tweet.replace("\n"," ")
    tweet = tweet.replace("\t"," ")
    docno = re.findall(r'<DOCNO>2011(\S{32}).*?</DOCNO>',tweet)
    text  = re.findall(r'<TEXT>.*?2011 (.*?)</TEXT>',tweet)
    tweet = zip(docno,text)
    document = []
    for t in tweet:
        if is_alphabet(t[1]):
            document.append(t)
    return document

def saveTweet(begin,end):
    file = open("/mnt/hgfs/data/2011/t/"+str(begin)+".txt","w+")
    for i in range(begin,end):
        document = readFile(path[i])
        for item in document:
            if len(item[1])>20:
                file.writelines(["%s %s\n" % (item[0],item[1])])
        print path[i]
    file.close()


path = glob.glob(r"/mnt/hgfs/data/cc/*/*.tweet")

import multiprocessing
mpool = multiprocessing.Pool(processes=8)
length = 30
num = len(path)/length
for i in xrange(num):
    begin = i*length
    end = (i+1)*length
    mpool.apply_async(saveTweet, (begin,end))

mpool.apply_async(saveTweet, ((i+1)*length,len(path)))
mpool.close()
mpool.join()
print "Sub-process(es) done."



# saveTweet(0,50)
