import math
import numpy as np
import cv2

FILE_PATH = "c:/Users/wcoh/AI_Study/woong/practice/images/"
FILE_SOURCE = "source"  # ORIGINAL IMAGE
FILE_TARGET = "target"  # REFERENCE IMAGE
FILE_EXTENTION = ".jpg"

img_source = cv2.imread(FILE_PATH + FILE_SOURCE +
                        FILE_EXTENTION, cv2.IMREAD_COLOR)
img_target = cv2.imread(FILE_PATH + FILE_TARGET +
                        FILE_EXTENTION, cv2.IMREAD_COLOR)
h_source, w_source, channel = img_source.shape
h_target, w_target, channel2 = img_target.shape

# Mean

mean_source_b = 0
mean_source_g = 0
mean_source_r = 0

mean_target_b = 0
mean_target_g = 0
mean_target_r = 0

for j in range(0, h_source):
    for i in range(0, w_source):
        mean_source_b = mean_source_b + img_source.item(j, i, 0)
        mean_source_g = mean_source_g + img_source.item(j, i, 1)
        mean_source_r = mean_source_r + img_source.item(j, i, 2)

for j in range(0, h_target):
    for i in range(0, w_target):
        mean_target_b = mean_target_b + img_target.item(j, i, 0)
        mean_target_g = mean_target_g + img_target.item(j, i, 1)
        mean_target_r = mean_target_r + img_target.item(j, i, 2)

mean_source_b = mean_source_b / (h_source * w_source)
mean_source_g = mean_source_g / (h_source * w_source)
mean_source_r = mean_source_r / (h_source * w_source)
mean_target_b = mean_target_b / (h_target * w_target)
mean_target_g = mean_target_g / (h_target * w_target)
mean_target_r = mean_target_r / (h_target * w_target)

# Std
var_source_b = 0
var_source_g = 0
var_source_r = 0

var_target_b = 0
var_target_g = 0
var_target_r = 0

for j in range(0, h_source):
    for i in range(0, w_source):
        var_source_b = var_source_b + \
            pow((img_source.item(j, i, 0) - mean_source_b), 2)
        var_source_g = var_source_g + \
            pow((img_source.item(j, i, 1) - mean_source_g), 2)
        var_source_r = var_source_r + \
            pow((img_source.item(j, i, 2) - mean_source_r), 2)

for j in range(0, h_target):
    for i in range(0, w_target):
        var_target_b = var_target_b + \
            pow((img_target.item(j, i, 0) - mean_target_b), 2)
        var_target_g = var_target_g + \
            pow((img_target.item(j, i, 1) - mean_target_g), 2)
        var_target_r = var_target_r + \
            pow((img_target.item(j, i, 2) - mean_target_r), 2)

var_source_b = math.sqrt(var_source_b / (h_source * w_source))
var_source_g = math.sqrt(var_source_g / (h_source * w_source))
var_source_r = math.sqrt(var_source_r / (h_source * w_source))

var_target_b = math.sqrt(var_target_b / (h_target * w_target))
var_target_g = math.sqrt(var_target_g / (h_target * w_target))
var_target_r = math.sqrt(var_target_r / (h_target * w_target))

# color transfer
for j in range(0, h_source):
    for i in range(0, w_source):
        new_Val_b = (img_source.item(j, i, 0) - mean_source_b) * (var_target_b / var_source_b) + mean_target_b
        new_Val_g = (img_source.item(j, i, 1) - mean_source_g) * (var_target_g / var_source_g) + mean_target_g
        new_Val_r = (img_source.item(j, i, 2) - mean_source_r) * (var_target_r / var_source_r) + mean_target_r
        
        if(new_Val_b > 255):
            new_Val_b = 255
        if(new_Val_g > 255):
            new_Val_g = 255
        if(new_Val_r > 255):
            new_Val_r = 255

        if(new_Val_b < 0):
            new_Val_b = 0
        if(new_Val_g < 0):
            new_Val_g = 0
        if(new_Val_r < 0):
            new_Val_r = 0

        img_source.itemset((j, i, 0), new_Val_b)
        img_source.itemset((j, i, 1), new_Val_g)
        img_source.itemset((j, i, 2), new_Val_r)

# output

cv2.imshow("source image", img_source)
cv2.imshow("target image", img_target)
cv2.waitKey(0)
