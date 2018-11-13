# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:57:59 2018

@author: Anthony
"""

import urllib.request
response = urllib.request.urlopen('https://shop.lululemon.com/')
html = response.read()


from http.cookiejar import CookieJar
import json
url = 'https://shop.lululemon.com/p/women-crops/Align-Crop-21'
req = urllib.request.Request(url, None, {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8','Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3','Accept-Encoding': 'gzip, deflate, sdch','Accept-Language': 'en-US,en;q=0.8','Connection': 'keep-alive'})

cj = CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
response = opener.open(req)
doc = response.read()
from bs4 import BeautifulSoup
soup = BeautifulSoup(doc, 'html.parser')
name_box = soup.find('div')



html = json.loads(response.content.decode('utf-8'))


response = urllib.request.urlopen('https://shop.lululemon.com/p/women-crops/Align-Crop-21')


from lxml import html
import requests

page = requests.get('https://shop.lululemon.com/p/women-crops/Align-Crop-21')
tree = html.fromstring(page.content)


from html.parser import HTMLParser
parser = HTMLParser()


import urllib.request
fp = urllib.request.urlopen("https://www.pixiv.net/")
mybytes = fp.read()

mystr = mybytes.decode("utf8")
fp.close()

print(mystr)



thing = urllib.request.HTTPRedirectHandler()
thing2 = urllib.request.HTTPCookieProcessor()
opener = urllib.request.build_opener(thing, thing2)
url = 'https://shop.lululemon.com/p/women-crops/Align-Crop-21'
page = opener.open(url)
string = page.read()