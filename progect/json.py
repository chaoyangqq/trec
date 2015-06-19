import json,re

def saveSR(feature,name):
    path = "/mnt/hgfs/data/2011/feature/"+str(name)+".txt"
    file = open(path, 'w+')
    for f in feature:
        f = f.encode('ascii', 'ignore').decode('ascii')
        file.write(f+"\n")
    file.close()


path = "/mnt/hgfs/data/2011/feature/MB01Noutput.json"
input = open(path, 'r').read()
s = json.loads(input)
searchEngine=['google','bing']
feature =[[] for i in range(len(searchEngine))]


for index,search in enumerate(searchEngine):
    for i in range(len(s)):
        if s[i]['search_engine_name'] == search:
            for result in s[i]['results']:
                feature[index].append(result['title']+result['snippet'])
    saveSR(feature[index],"n"+search)        



