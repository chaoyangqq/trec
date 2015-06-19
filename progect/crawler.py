#coding=utf-8
import urllib,re
from bs4 import BeautifulSoup

def getHtml(url):
    page = urllib.urlopen(url)
    html = page.read()
    return html

def gettwitter(query):
	html = getHtml("https://twitter.com/search?q="+query+"&src=typd")
	#html = open('/root/sample_data/1.html','r').readlines()
	#print html
	soup = BeautifulSoup(html)
	twits = soup.find_all("p",class_="TweetTextSize")
	twitters=[]
	for t in twits:
		dr = re.compile(r'<[^>]+>',re.S)
		twitters.append(dr.sub('',str(t))+"\n")

	return twitters


airline mergers

Description:
Information on potential or completed mergers in the airline industry.

twitters = []
for t in query:
	twitters.extend(gettwitter(t))


f = open('/root/sample_data/twitter.txt','a')
f.writelines(twitters)
f.close()

# text = nltk.word_tokenize(narr[0])
# print nltk.pos_tag(text)

#twitter_stoplist = ["relevant","gave","wants"]
# stoplist = nltk.corpus.stopwords.words('english')

# for sentence in relevant[1]:
#     for word in sentence.split(): 
#         if word.lower() not in stoplist:
#             feature.append(word.lower())

# feature = list(set(feature))

# print wikipedia.summary(sentence)


        #print word
    # try:
    # 	wiki = wikipedia.search("word")
    # feature.extend(wiki)


#extractor.filter = extract.permissiveFilter

