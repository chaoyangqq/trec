from google import search
import urllib
dsc=[]
url=[]
for u in search('cool', stop=20):
    url.append(u)
    page = urllib.urlopen(url)  
    charset = page.headers['Content-Type'].lower().split("charset=")[1]  
    content = page.read().decode(charset, "ignore").encode("utf-8",'ignore')
    soup = BeautifulSoup(content)
    description = soup.find(attrs={"name":"description"})['content']
    keywords = soup.find(attrs={"name":"keywords"})['content']
    dsc.append(description + keywords)

from google import search
for url in search('cool', stop=20):
    print(url)



try:
  gs = GoogleSearch("quick and dirty")
  gs.results_per_page = 50
  results = gs.get_results()
  for res in results:
    print res.title.encode("utf8")
    print res.desc.encode("utf8")
    print res.url.encode("utf8")
    print
except SearchError, e:
  print "Search failed: %s" % e


from googlefinder.google import finder
x = finder()
x.Search(['airline mergers'])
