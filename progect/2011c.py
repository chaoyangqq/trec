import os,re,thread,time
import glob
import cchardet as chardet
import profile 

def is_alphabet(rawdata):
    type=chardet.detect(rawdata)
    if type['encoding']=='ASCII':
        return True
    else:
        return False

def readFile(path):
    tweet = open(path,"r").read()
    tweet = tweet.replace("\n"," ")
    tweet = tweet.replace("\t"," ")
    docno = re.findall(r'<DOCNO>.*?\|(\d{17}).*?</DOCNO>',tweet)
    text  = re.findall(r'<TEXT>.*?2011 (.*?)</TEXT>',tweet)
    tweet = zip(docno,text)
    document = []
    for t in tweet:
        if is_alphabet(t[1]) and len(t[1])>15:
            document.append(t)
    return document

def gettext(documents):
    twitter_stoplist = ["what's","it's","they'd"]
    stoplist = nltk.corpus.stopwords.words('english')+twitter_stoplist
    texts = []
    for document in documents:
        line=[]
        d = re.sub(r'[^a-zA-Z]+'," ", document)
        d = re.sub(r'\s+', ' ',d)
        for word in d.strip().split():
            word = word.lower()
            if word not in stoplist:
                if '@' in word:
                    continue
                    #word="USER"
                if len(word)>15:
                    continue
                    # word="LW"
                elif len(word)<3:
                    continue
                #    word="SW"
                    
                line.append(word)
        texts.append(line)
    return texts

def saveTweet(begin,end):
    file = open("/mnt/hgfs/data/2011/"+str(begin)+".txt","w+")
    for i in range(begin,end):
        document = readFile(path[i])
        for item in document:
            file.writelines(["%s %s\n" % (item[0],item[1])])
        print path[i]
    file.close()


path = glob.glob(r"/mnt/hgfs/data/cc/*/*.tweet")
saveTweet(0,len(path))


len(path)
length = 50





# len(path)
# length = 50
# num = len(path)/length


# try:
#     for i in range(len(path)/length): 
#         begin = i*length
#         end = (i+1)*length
#         thread.start_new_thread(saveTweet,(begin,end))
# except:
#     print "Error: unable to start thread"

# saveTweet(0,len(path))
# saveTweet(0,2)

# profile.run("saveTweet(0,2)")

# len(path)
# length = 50
# num = len(path)/length
# Parameter =  [(x,y) for x in range(0,num),y in range(1,num+1)]

# import threadpool
# pool = threadpool.ThreadPool(10)  #建立线程池，控制线程数量为10
# requests = threadpool.makeRequests(saveTweet,Parameter)
# [pool.putRequest(req) for req in requests]
# pool.wait()











