"""
#wordcloud using Twitter
from collections import Counter
import webbrowser
from konlpy.tag import Twitter
import pytagcloud

f = open('C:\\Users\\sumin0422\\AI_Study\\sumin\\toystory.txt')
data = f.read()
 
nlp = Twitter() 
nouns = nlp.nouns(data)
 
count = Counter(nouns)
tags2 = count.most_common(40)
taglist = pytagcloud.make_tags(tags2, maxsize=80)
pytagcloud.create_tag_image(taglist, 'wordcloud.jpg', size=(600, 400), fontname='sumin', rectangular=False)

webbrowser.open("wordcloud.jpg)
 
"""
#! /usr/bin/python2.7
# -*- coding: utf-8 -*-
""" wordcloud using Hannanum"""
from collections import Counter
import urllib
import random
import webbrowser

from konlpy.tag import Hannanum
from lxml import html
import pytagcloud # requires Korean font support
import sys


r = lambda: random.randint(0,255)
color = lambda: (r(), r(), r())

def get_tags(text, ntags=50, multiplier=10):
    h = Hannanum()
    nouns = h.nouns(text)
    count = Counter(nouns)
    return [{ 'color': color(), 'tag': n, 'size': c*multiplier }\
                for n, c in count.most_common(ntags)]

def draw_cloud(tags, filename, fontname='sumin', size=(700, 500)):
    pytagcloud.create_tag_image(tags, filename, fontname=fontname, size=size)
    webbrowser.open(filename)


#bill_num = '1918854'
#text = get_bill_text(bill_num)
f=open('C:\\Users\\sumin0422\\AI_Study\\sumin\\toystory.txt')
data = f.read()
tags = get_tags(data)
print(tags)
draw_cloud(tags, 'wordcloud.png')

