#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from requests.auth import HTTPBasicAuth

## Last data from jarvisdata
url1 = 'http://jarvisglobal.co:5050/lastdata/exchange=binance&symbol=btcperp'
data1 = requests.get(url1, auth=HTTPBasicAuth('JarvisUsers', 'Jarvisglobalusers20'))
print('last data:', data1.json())
print('--------------')


# In[3]:


## Last number data from jarvisdata
url2 = 'http://jarvisglobal.co:5050/data/exchange=binance&symbol=btcperp&depth=10'
data2 = requests.get(url2, auth=HTTPBasicAuth('JarvisUsers', 'Jarvisglobalusers20'))
print('last10 data:' , data2.json())
print('--------------')


# In[8]:


## Period data from jarvisdata
url3 = 'http://jarvisglobal.co:5050/period/exchange=binance&symbol=btcperp&start=2020-03-12-15:00:00&end=2020-03-12-15:05:49'
data3 = requests.get(url3, auth=HTTPBasicAuth('JarvisUsers', 'Jarvisglobalusers20'))
print('period data:' , data3.json())
print('--------------')


# In[9]:


## Last data from jarvisdatabis
url4 = 'http://jarvisglobal.co:5050/lastfunding/exchange=Deribit'
data4 = requests.get(url4, auth=HTTPBasicAuth('JarvisUsers', 'Jarvisglobalusers20'))
print('last funding:' , data4.json())
print('--------------')


# In[10]:


## Last number data from jarvisdatabis
url5 = 'http://jarvisglobal.co:5050/funding/exchange=Deribit&depth=10'
data5 = requests.get(url5, auth=HTTPBasicAuth('JarvisUsers', 'Jarvisglobalusers20'))
print('last10 funding:' , data5.json())
print('--------------')


# In[13]:


## Period data from jarvisdatabis
url6 = 'http://jarvisglobal.co:5050/periodfunding/exchange=Deribit&start=2020-03-12_16:05:53&end=2020-03-12_16:08:44'
data6 = requests.get(url6, auth=HTTPBasicAuth('JarvisUsers', 'Jarvisglobalusers20'))
print('period funding:' , data6.json())
print('--------------')


# In[ ]:




