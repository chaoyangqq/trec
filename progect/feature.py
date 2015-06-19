import nltk,re,string,pprint
import wikipedia
import urllib,re
from googlefinder.google import finder
from bs4 import BeautifulSoup
import time
from topia.termextract import extract
import commands
import string
from TwitterSearch import *

#------------------remove targes------------------
#fen ge ju zi 
def tsplit(string, delimiters):
    delimiters = tuple(delimiters)
    stack = [string,]
    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i+j, _substring)
    return stack

def getHtml(url):
    page = urllib.urlopen(url)
    html = page.read()
    return html

def gettwitter(query):
    try:
        tso = TwitterSearchOrder()
        tso.set_language('en')
        tso.set_locale('en')
        tso.set_keywords([query])
        url = "https://twitter.com/search"+tso.create_search_url()
        print url
    except TwitterSearchException as e:
        print(e)
    html = getHtml(url)
    soup = BeautifulSoup(html)
    twits = soup.find_all("p",class_="TweetTextSize")
    twitters=[]
    for t in twits:
        dr = re.compile(r'<[^>]+>',re.S)
        replacedStr = dr.sub('',str(t))
        replacedStr = re.sub(r"([a-zA-z]+://.*$)", "url", replacedStr)
        twitters.append(replacedStr+"\n")
    return twitters


#----------began process features-------------------------------
html = open("/usr/python/progect/profileSamples.txt",'r+').read()
html = html.replace("\n"," ")
html = html.replace("(i.e.,","#")

num =re.findall('<num> Number: (.*?)\s*<title>',html)
title =re.findall('<title>\s(.*?)\s*<desc>',html)
desc = re.findall('<desc>\s(.*?)\s*<narr>',html)  
narr = re.findall('<narr> Narrative:\s*(.*?)\s*</top',html) 

relevant = [[] for i in range(len(narr))]
nrelevant = [[] for i in range(len(narr))]

for i in range(len(narr)):
    sentences = tsplit(narr[i], (',', ':', '.', ';'))
    for sentence in sentences:
        if "not" in sentence.split():
            nrelevant[i].append(sentence)  
        else:
            relevant[i].append(sentence)


extractor = extract.TermExtractor()
extractor.filter = extract.DefaultFilter(singleStrengthMinOccur=1)

for i in range(len(num)):
    pf = []
    pfeature = []
    for sentence in relevant[i]:
        pf.append(extractor(sentence))
    for line in pf:
        for t in line:
            pfeature.append(t[0])
    pfeature.append(title[i])
    pfeature = list(set(pfeature))
    print num[i]+"positive:"
    print pfeature
    f = open('/usr/python/progect/feature/'+num[i]+'fp.txt','w+')
    for p in pfeature:
        f.write(p+"\n")
    f.close()
    if len(nrelevant[i]) != 0:
        nf = []
        nfeature = []
        for sentence in nrelevant[i]:
            nf.append(extractor(sentence))
        for line in nf:
            for t in line:
                nfeature.append(t[0]) 
        nfeature = list(set(nfeature))
        print num[i]+"negitive:"
        print nfeature
        f = open('/usr/python/progect/feature/'+num[i]+'fn.txt','w+')
        for p in nfeature:
            f.write(p+"\n")
        f.close()





commands.getstatusoutput('GoogleScraper -m http -n 50 -p 10 --keyword-file /usr/python/progect/p.txt --num-workers 50 --search-engines "google,bing,yahoo,Baidu" --output-filename /usr/python/progect/output.json -v2')




for i in range(len(num)):
    f = open('/mnt/hgfs/data/2011/feature/'+num[i]+'twts.txt','w')
    twitter = gettwitter(title[i])
    while len(twitter) < 15:
        q = ''
        tit = title[0].strip().split(' ')
        for t in tit[:-1]:
            q += t+" "
        twitter.extend(gettwitter(q))
    f.writelines(twitter)
    f.close()


mb = "BBC World Service staff cuts"
nkey = []
for k in mb:
    nkey.extend(wikipedia.search("k"))
nkey = list(set(nkey))
f = open('/mnt/hgfs/data/2011/feature/MB01nkey.txt','w')
for u in nkey:
    if len(u)>2:
        f.write(u+"\n")

f.close()
