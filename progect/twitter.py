from TwitterSearch import *
import time
try:
    tso = TwitterSearchOrder()
    html = open("/usr/python/progect/profileSamples.txt",'r+').read()
    tso.set_keywords(['foo', 'bar'])
    ts = TwitterSearch(
        consumer_key='ooRo1YonXsAvS7RsgNYSCdxws',
        consumer_secret='0zX2rbeNiz5qxnUC9MB9uUErTB9EGDSG4OtU57tuPCsNI5u6M2', 
        access_token='3107850807-zCmoMZSwaRCvjuw9wybqBoyrGIDIkRoF7RU2am3',
        access_token_secret='FrHzLCkIHTXk8h62E5Pyl5ISNfk6ux3hvJDQiiEt9JqRE'
    )
    def my_callback_closure(current_ts_instance): # accepts ONE argument: an instance of TwitterSearch
        queries, tweets_seen = current_ts_instance.get_statistics()
        if queries > 0 and (queries % 5) == 0: # trigger delay every 5th query
            time.sleep(60) # sleep for 60 seconds
    for tweet in ts.search_tweets_iterable(tso, callback=my_callback_closure):
        print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )
except TwitterSearchException as e:
    print(e)


#----------get feature twitter-------------------------------

# twitters = []
# for t in pfeature:
#     twitters.extend(gettwitter(t))
#     time.sleep(random.randint(3, 6))
#     print(t)

# f = open('/usr/python/progect/'+num+'twitter.txt','w')
# f.writelines(twitters)
# f.close()

#----------get feature google-------------------------------
# result=[]
# x = finder()
# for t in pfeature:
#     try:
#         result.extend(x.Search([t]))
#         time.sleep(random.randint(3, 6))
#         print(t)
#     except:
#         print '\nSome error/exception occurred.'+t
    

# # f = open('/root/sample_data/p.txt','w')
# # for i in range(len(result)):
# #     t = result[i][1].encode('ascii', 'replace').replace("\n"," ")
# #     f.write(t+"\n")

# f.close()


from TwitterSearch import TwitterSearchOrder, TwitterSearchException
import urllib2
try:
    tso = TwitterSearchOrder()
    tso.set_language('en')
    tso.set_locale('en')
    tso.set_keywords(['airline mergers'])
    url = "https://twitter.com/search"+tso.create_search_url()
    print url
    response = urllib2.urlopen('http://www.baidu.com/')  
    html = response.read()  
    print html 
    
except TwitterSearchException as e:
      print(e)

tso2 = TwitterSearchOrder()
tso2.set_search_url(querystr + '&result_type=mixed&include_entities=true')
tso2.set_locale('en')
print(tso2.create_search_url())

    
tso = TwitterSearchOrder()
keywords = open("/usr/python/progect/feature/"+filename+".txt",'r+').read()
keywords = keywords.split("\n") 
tso.set_keywords("airline mergers")
ts = TwitterSearch(
    consumer_key='ooRo1YonXsAvS7RsgNYSCdxws',
    consumer_secret='0zX2rbeNiz5qxnUC9MB9uUErTB9EGDSG4OtU57tuPCsNI5u6M2', 
    access_token='3107850807-zCmoMZSwaRCvjuw9wybqBoyrGIDIkRoF7RU2am3',
    access_token_secret='FrHzLCkIHTXk8h62E5Pyl5ISNfk6ux3hvJDQiiEt9JqRE'
)


ts.set_supported_languages(tso)
tso.set_language('en')
def my_callback_closure(current_ts_instance): # accepts ONE argument: an instance of TwitterSearch
    queries, tweets_seen = current_ts_instance.get_statistics()
    if queries > 0 and (queries % 5) == 0: # trigger delay every 5th query
        time.sleep(60) # sleep for 60 seconds
f = open('/usr/python/progect/feature/filename.txt','w+')
sleep_for = 60 # sleep for 60 seconds
last_amount_of_queries = 0 # used to detect when new queries are done
for tweet in ts.search_tweets_iterable(tso, callback=my_callback_closure):
    print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )
    temp = re.findall('\\x\w{3})'," ",tweet['text']) 
    f.write(tweet['text'].replace("\n"," ")+"\n")

f.close()



from twython import Twython
twitter = Twython('ooRo1YonXsAvS7RsgNYSCdxws','0zX2rbeNiz5qxnUC9MB9uUErTB9EGDSG4OtU57tuPCsNI5u6M2', '3107850807-zCmoMZSwaRCvjuw9wybqBoyrGIDIkRoF7RU2am3','FrHzLCkIHTXk8h62E5Pyl5ISNfk6ux3hvJDQiiEt9JqRE')
result = twitter.search(q='airline mergers')
f = open('/usr/python/progect/feature/t1','w')
for tws in result['statuses']:
    t = tws
    f.write(t+"\n")

f.close()


tso = TwitterSearchOrder()
tso.set_language('en')
tso.set_locale('en')
tso.set_keywords(['airline mergers'])
print(tso.create_search_url())

tso2 = TwitterSearchOrder()
    tso2.set_search_url(querystr + '&result_type=mixed&include_entities=true')
    tso2.set_locale('en')
    print(tso2.create_search_url())


tso = TwitterSearchOrder()
keywords = open("/usr/python/progect/feature/"+filename+".txt",'r+').read()
keywords = keywords.split("\n") 
tso.set_keywords("airline mergers")
ts = TwitterSearch(
    consumer_key='ooRo1YonXsAvS7RsgNYSCdxws',
    consumer_secret='0zX2rbeNiz5qxnUC9MB9uUErTB9EGDSG4OtU57tuPCsNI5u6M2', 
    access_token='3107850807-zCmoMZSwaRCvjuw9wybqBoyrGIDIkRoF7RU2am3',
    access_token_secret='FrHzLCkIHTXk8h62E5Pyl5ISNfk6ux3hvJDQiiEt9JqRE'
)


ts.set_supported_languages(tso)
tso.set_language('en')
def my_callback_closure(current_ts_instance): # accepts ONE argument: an instance of TwitterSearch
    queries, tweets_seen = current_ts_instance.get_statistics()
    if queries > 0 and (queries % 5) == 0: # trigger delay every 5th query
        time.sleep(60) # sleep for 60 seconds
f = open('/usr/python/progect/feature/filename.txt','w+')
sleep_for = 60 # sleep for 60 seconds
last_amount_of_queries = 0 # used to detect when new queries are done
for tweet in ts.search_tweets_iterable(tso, callback=my_callback_closure):
    print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )
    temp = re.findall('\\x\w{3})'," ",tweet['text']) 
    f.write(tweet['text'].replace("\n"," ")+"\n")

f.close()


except TwitterSearchException as e:
    print(e)


gettwitters(num[1]+'fp')



