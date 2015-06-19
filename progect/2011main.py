__author__ = "ROOT"
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from gensim import matutils,corpora, models, similarities
from pprint import pprint
import nltk,re,os
import cchardet as chardet
import json
import string
import numpy
import math
import glob
import multiprocessing
from operator import itemgetter


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def is_alphabet(rawdata):
    type=chardet.detect(rawdata)
    if type['encoding']=='ASCII':
        return True
    else:
        return False

def get_URL(rawdata):
    m = re.findall('(https{0,1}://t\.co/\w{10})',rawdata)  
    return m

def get_URL(rawdata):
    #m = re.findall('(https{0,1}.?\s)',rawdata)  
    m = re.findall('(https{0,1}://t\.co/\w{10})',rawdata) 
    return m

def get_URLInfo(rawdata):
    url = get_URL(rawdata)
    for a in url:
        html1 = str(urllib.urlopen(str(a)))
        print html1
        t = re.search('(<meta.*property="og:type".*content="(.*)">)',html1)
        u = re.search('(<meta.*property="og:url".*content="(.*)">)',html1)
        d = re.search('(<meta.*property="og:title".*content="(.*)">)',html1)
        print t,u,d

english_punctuations = [',', ';', '?', '(', ')', '[', ']', '&', '!', '*', '$', '%','\n','...','..','.',':','./']
punctuation = string.punctuation+string.whitespace  #'0123456789'string.digits'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'string.whitespace'\t\n\x0b\x0c\r '



def predocument(number):
    fname = "/mnt/hgfs/data/2011/feature/"+str(number)+".txt"
    f = open(fname,'r+')
    documents=[]
    with open(fname) as f:
        for line in f:
            if is_alphabet(line) and len(line)>15:
                for url in get_URL(line):
                    line=line.replace(str(url),"url")
                documents.append(line)
    documents = list(set(documents))
    return documents


def gettext(number):
    twitter_stoplist = ["what's","it's","they'd"]
    stoplist = nltk.corpus.stopwords.words('english')+twitter_stoplist
    texts = []
    documents = predocument(number)
    for document in documents:
        line=[]
        d = re.sub(r'[^a-zA-Z]+'," ", document)
        d = re.sub(r'\s+', ' ',d)
        for word in d.strip().split():
            word = word.lower()
            if word not in stoplist and '@' not in word and len(word)<15 and len(word)>3:
                word = nltk.PorterStemmer().stem(word)
                line.append(word)
        texts.append(line)
    return texts


###############################利用非负矩阵提取兴趣特征############################
interest = gettext('MB01twts')
positive = gettext('google')
negative = gettext('ngoogle')
sorted(positive,key=len)
sorted(negative,key=len)
positive = positive[-len(interest):]
negative = negative[-len(interest):]


dictionary = corpora.Dictionary(interest)
dictionary.add_documents(positive)
dictionary.add_documents(negative)

pc = [dictionary.doc2bow(t) for t in positive]
nc = [dictionary.doc2bow(t) for t in negative]
ic = [dictionary.doc2bow(t) for t in interest]

# tfidf = models.TfidfModel(dc)
# dc = tfidf[dc]


dimension = len(dictionary)
p = matutils.corpus2dense(pc,dimension)
n = matutils.corpus2dense(nc,dimension)
x = matutils.corpus2dense(ic,dimension)


#x.shape[0]:296
#x.shape[1]:11
k=x.shape[1]/2
u = numpy.random.rand(x.shape[0],k)+1
h = numpy.random.rand(x.shape[1],k)+1
a = numpy.random.rand(x.shape[0],k)+1
b = numpy.random.rand(x.shape[0],k)+1

for i in range(1000):
    utempa = (numpy.dot(x,h)+a)
    utempb = (numpy.dot(numpy.dot(u,numpy.transpose(h)),h)+b)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            u[i][j]=(u[i][j]*utempa[i][j]+1.0)/(utempb[i][j]+1.0)
    htempa = (numpy.dot(numpy.transpose(x),u)+numpy.dot(numpy.transpose(p),a)+numpy.dot(numpy.transpose(n),b))
    htempb = (numpy.dot(numpy.dot(h,numpy.transpose(u)),u)+numpy.dot(numpy.dot(h,numpy.transpose(a)),a)+numpy.dot(numpy.dot(h,numpy.transpose(b)),b))
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            h[i][j]=(h[i][j]*htempa[i][j]+1.0)/(htempb[i][j]+1.0)
    atempa = (numpy.dot(p,h)+u)
    atempb = (numpy.dot(numpy.dot(a,numpy.transpose(h)),h)+a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i][j]=(a[i][j]*atempa[i][j]+1.0)/(atempb[i][j]+1.0)
    btempa = (numpy.dot(n,h)+u)
    btempb = (numpy.dot(numpy.dot(b,numpy.transpose(h)),h)+b)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            b[i][j]=(b[i][j]*btempa[i][j]+1.0)/(btempb[i][j]+1.0)



###############################处理文档############################
def getRank(fname,numpyfname):
    twitter_stoplist = ["what's","it's","they'd"]
    stoplist = nltk.corpus.stopwords.words('english')+twitter_stoplist
    document = []
    with open(fname) as f:
        for line in f:
            tl = []
            tl.append(line[:32])#添加标号
            line=re.sub(r"(http:.*?\s)","url" ,line[33:])
            d = re.sub(r'[^a-zA-Z]+'," ", line[33:])#去除非英文字符
            d = re.sub(r'\s+',' ',d)#合并多余空格
            for word in d.split():
                word = word.lower()
                if word not in stoplist and '@' not in word and len(word)<15 and len(word)>3:
                    tl.append(word)
            document.append(tl)
    print fname+" read complate"
    dc = [dictionary.doc2bow(t) for t in document]
    d = matutils.corpus2dense(dc,dimension)
    for i in range(len(d[0])):
        tsum = sum(d[:,i])
        if tsum != 0:
            for j in range(len(d)):
                if d[j][i]/tsum > 0.6:
                    d[j][i] = 2
    r2 = numpy.dot(numpy.transpose(u),d)
    score2 =numpy.array([sum(r2[:,i]) for i in range(len(r2[0]))])
    x = [document[i][0] for i in range(len(document))]
    l2 = zip(score2,x)
    # l2 = sorted(l2, key=itemgetter(0,1),reverse=True)
    # if len(l2)>10000:
    #     l2=l2[:10000]
    print fname+" process complate"
    # tempf = "/mnt/hgfs/data/2011/numpy/"+str(numpyfname)+".npy"
    # print tempf    
    # numpy.save(tempf, d)
    return l2


result_list = []
def log_result(result):
    result_list.extend(result)

path = glob.glob(r"/mnt/hgfs/data/2011/t/*.txt")
pool = multiprocessing.Pool(processes=8)
for index,p in enumerate(path):
    pool.apply_async(getRank,(p,index), callback = log_result)

pool.close()
pool.join()
print "Sub-process(es) done."


result = sorted(result_list, key=itemgetter(0,1),reverse=True)
finalresult = [a[1] for a in result]
finalresult = [a[1][15:] for a in result]

f =open("/mnt/hgfs/data/2011/result/1.txt",'w+')
for i in result[:10000]:
    f.write(i[1]+"\n")

f.close()


answer=[] 
answerScore =[]
with open("/mnt/hgfs/data/2011/microblog11-qrels.txt") as f:
    for line in f:
        if line[0] == '1' and line[1]==' ':
            answer.append(line[4:21])
            answerScore.append(line[22])

correctRate =list(set(answer) & set(finalresult))


for i in range(len(correctRate)):
    print correctRate[i]+","+str(finalresult.index(correctRate[i]))+","+answerScore[answer.index(correctRate[i])]


lcyr = [(correctRate[i],finalresult.index(correctRate[i]),answerScore[answer.index(correctRate[i])]) for i in range(len(correctRate))]
lcyr1 =sorted(lcyr, key=itemgetter(1),reverse=True) 
lcyr1[:10]
lcyr2 =sorted(lcyr, key=itemgetter(2))
lcyr2[:10]

finalresult.index("34553453812387840")




import matplotlib.pyplot as plt
pl1 = numpy.zeros(len(l1),dtype=numpy.int)
pl2 = numpy.zeros(len(l2),dtype=numpy.int)
for i in range(len(l1)):
    pl1[i] = l1[i][1]
    pl2[i] = l2[i][1]


plt.plot(pl1)
plt.plot(pl2)
plt.xlabel('Similarity')  
plt.ylabel('numbers')  
plt.show()   

tmp1= numpy.where(score1>numpy.max(score1)-30)
tmp2= numpy.where(score2>numpy.max(score2)-2)

for t in tmp1[0]:
    print documents[t]
for t in tmp2[0]:
    print documents[t]

numpy.savetxt("/root/sample_data/u.txt",u)

#------------------------------------------------------------------------
r1 = numpy.dot(numpy.transpose(x),d)
score1 =numpy.array([sum(r1[:,i]) for i in range(len(r1[0]))])
l1 = zip(score1,range(len(score1)))
l1 = sorted(l1, key=itemgetter(0,1),reverse=True) 
l1 = l1[:100]
#-------------------------------------------------------------------------
###############################利用非负矩阵提取兴趣特征end###############################
dictionary = corpora.Dictionary(texts)
#dictionary.filter_extremes(2,0.01,200000)
dictionary.add_documents(texts)

corpus = [dictionary.doc2bow(t) for t in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
#lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=100)

corpus_lsi = lsi[corpus_tfidf]

cluster = [[] for i in range(100)]
i=0
for doc in corpus_lsi: 
    kind = 0
    posible=0
    for p in doc:
        if p[1] > posible:
            kind = p[0]
            posible = p[1]
    cluster[kind].append(texts[i])
    i+=1



##############
################


v = dictionary.values()
n = len(texts)
prior1=[]
prior2=[]
condprob1 = numpy.zeros((len(v),100),dtype=numpy.float64)
condprob2 = numpy.zeros((len(v),100),dtype=numpy.float64)


for i in range(100):
    textc = []
    nc = len(cluster[i])    
    for line in cluster[i]:
        for w in line:
            textc.append(w)
    tct0 =dictionary.doc2bow(textc)
    np = numpy.array(tct0)
    prior1.append(nc/n) 
    tct1 = len(textc)+len(v)
    ii=0
    for j in xrange(len(v)):
        if ii<len(np) and np[ii][0]==j:
            condprob1[j][i]=np[ii][1]/tct1
            ii+=1
        else:
            condprob1[j][i]=1/tct1


for i in range(100):
    textc = []
    nc = 0
    for ii in range(100):
        if ii==i:
            continue
        nc += len(cluster[ii])    
        for line in cluster[ii]:
            for w in line:
                textc.append(w)
    prior2.append(nc/n) 
    tct0 =dictionary.doc2bow(textc)
    np = numpy.array(tct0)
    tct1 = len(textc)+len(v)
    ii=0
    for j in xrange(len(v)):
        if ii<len(np) and np[ii][0]==j:
            condprob2[j][i]=np[ii][1]/tct1
            ii+=1
        else:
            condprob2[j][i]=1/tct1


#numpy_matrix = matutils.corpus2dense(tct0,len(tct0))

textst = gettext("test")
W = [dictionary.doc2bow(t) for t in textst]
score=numpy.zeros((len(W),100),dtype=numpy.float64)
for i in range(len(W)):
    for j in range(100):
        score[i][j]=math.log(prior1[1])
        for w in W[i]:
            score[i][j]+=math.log(condprob1[w[0]][j])



tt = predocument("test")
Wcluster = [[] for i in range(100)]
kkind=[]
for i in range(len(score)):
    kind = 0
    posible= -10000
    for j in range(len(score[i])):
        if score[i][j] > posible:
            posible = score[i][j]
            kind = j
    kkind.append(kind) 
    Wcluster[kind].append(tt[i])



##############################


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(documents)
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)


#**********************************
import profile 
profile.run("getDocument(1600)")
#**********************************