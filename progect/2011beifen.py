# def getPath(rootDir):
#     p = [] 
#     for lists in os.listdir(rootDir): 
#         path = os.path.join(rootDir, lists) 
#         if os.path.isdir(path): 
#             getPath(path)
#         else:
#             print path
#             p.append(path)
#     return p
# path = getPath("/mnt/hgfs/data/cc") 
# from os import listdir
# from os.path import isfile, join
# onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

# with open("/mnt/hgfs/data/1.txt", 'wb') as f:
#     pickle.dump(document, f)

# with open(/"mnt/hgfs/data/1.txt", 'rb') as f:
#     document = pickle.load(f)

# f = open("/mnt/hgfs/data/1.txt","w")
# l = document
# simplejson.dump(l,f)
# f.close()

# file_contents = simplejson.load(f)

# readFile("/mnt/hgfs/data/cc/20110123/20110123-000.tweet")
# path = "/mnt/hgfs/data/cc/20110123/20110123-000.tweet"
