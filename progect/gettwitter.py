from TwitterSearch import TwitterSearchOrder, TwitterSearchException
import urllib2,urllib,re
from bs4 import BeautifulSoup

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
        replacedStr = re.sub(r"([a-zA-z]+://\S*\s{0,1})", "url", replacedStr)
        twitters.append(replacedStr+"\n")
    return twitters


f = open('/mnt/hgfs/data/2011/feature/MB01twts.txt','w')
title = "BBC World Service staff cuts"
twitter = gettwitter(title)
while len(twitter) < 15:
    q = ''
    tit = title.strip().split(' ')
    for t in tit[:-1]:
        q += t+" "
    twitter.extend(gettwitter(q))

f.writelines(twitter)
f.close()


commands.getstatusoutput('GoogleScraper -m http -n 50 -p 10 --keywords "BBC World Service staff cuts" --num-workers 50 --search-engines "google,bing,yahoo" --output-filename /mnt/hgfs/data/2011/feature/output.json -v2')

GoogleScraper -m http -n 50 --keyword-file /mnt/hgfs/data/2011/feature/MB01pkey.txt --num-workers 10 --search-engines "google,bing" --output-filename /mnt/hgfs/data/2011/feature/MB01Poutput.json -v2
GoogleScraper -m http -n 10 --keyword-file /mnt/hgfs/data/2011/feature/MB01nkey.txt --num-workers 10 --search-engines "google,bing" --output-filename /mnt/hgfs/data/2011/feature/MB01Noutput.json -v2

