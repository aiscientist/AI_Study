#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

from collections import Counter
import urllib
import random
import webbrowser

from konlpy.tag import Hannanum
from lxml import html
import pytagcloud # requires Korean font support
import sys

if sys.version_info[0] >= 3:
    urlopen = urllib.request.urlopen
else:
    urlopen = urllib.urlopen


r = lambda: random.randint(0,255)
color = lambda: (r(), r(), r())

def get_text(txt_name):
    try:
        PATH_TEXT = "c:/Users/wcoh/AI_Study/woong/practice/"
        f = open(PATH_TEXT + txt_name, 'r')
    except FileNotFoundError as e:
        print(str(e))
    else:
        data = f.read()
        f.close()
        return data
    
def get_tags(text, ntags=50, multiplier=10):
    h = Hannanum()
    nouns = h.nouns(text)
    count = Counter(nouns)
    return [{ 'color': color(), 'tag': n, 'size': c*multiplier }\
                for n, c in count.most_common(ntags)]

def draw_cloud(tags, filename, fontname='GodicBold', size=(800, 600)):
    pytagcloud.create_tag_image(tags, filename, fontname=fontname, size=size)
    webbrowser.open(filename)

text = get_text("news.txt")
tags = get_tags(text)
print(tags)
draw_cloud(tags, 'wordcloud.png')
