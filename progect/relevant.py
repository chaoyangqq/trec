from googlefinder.google import finder
import pprint,re
x = finder()
result = x.Search(['airline',''])

f = open('/root/sample_data/p.txt','a')
for i in range(len(result)):
    t = result[i][1].encode('ascii', 'replace').replace("\n"," ")
    #t = result[i][1].decode('ascii', 'ignore')
    #t = re.findall('r(\\x\S{2})',result[i][1])
    f.write(t+"\n")

f.close()

print(result)