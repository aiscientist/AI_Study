import math
import numpy as np
import cv2

FILE_PATH = "c:/Users/wcoh/AI_Study/woong/practice/images/"
FILE_SOURCE = "source"  # ORIGINAL IMAGE
FILE_TARGET = "target"  # REFERENCE IMAGE
FILE_EXTENTION = ".jpg"

class colorTransfer:
    meanB = [0, 0]
    meanG = [0, 0]
    meanR = [0, 0]
    stdB = [0, 0]
    stdG = [0, 0]
    stdR = [0, 0]
    h = [0, 0]
    w = [0, 0]

    def limit(self, val):
        if val > 255:
            return 255
        if val < 0:
            return 0

    def __init__(self, source, target):
        self.source_image = cv2.imread(FILE_PATH + source +
                        FILE_EXTENTION, cv2.IMREAD_COLOR)
        self.h[0], self.w[0], ch = self.source_image.shape
        
        self.target_image = cv2.imread(FILE_PATH + target +
                        FILE_EXTENTION, cv2.IMREAD_COLOR)
        self.h[1], self.w[1], ch = self.target_image.shape

    def calMean(self, imgName):
        if imgName == "source":
            N = 0
            img = self.source_image
        elif imgName == "target":
            N = 1
            img = self.target_image
        
        for j in range(0, self.h[N]):
            for i in range(0, self.w[N]):
                self.meanB[N] = self.meanB[N] + img.item(j, i, 0)
                self.meanG[N] = self.meanG[N] + img.item(j, i, 1)
                self.meanR[N] = self.meanR[N] + img.item(j, i, 2)

        self.meanB[N] = self.meanB[N] / (self.h[N]*self.w[N])
        self.meanG[N] = self.meanG[N] / (self.h[N]*self.w[N])
        self.meanR[N] = self.meanR[N] / (self.h[N]*self.w[N])

    def calStd(self, imgName):
        if imgName == "source":
            N = 0
            img = self.source_image
        elif imgName == "target":
            N = 1
            img = self.target_image
        
        for j in range(0, self.h[N]):
            for i in range(0, self.w[N]):
                self.stdB[N] = self.stdB[N] + \
                    pow((img.item(j, i, 0) - self.meanB[N]), 2)
                self.stdG[N] = self.stdG[N] + \
                    pow((img.item(j, i, 1) - self.meanG[N]), 2)
                self.stdR[N] = self.stdR[N] + \
                    pow((img.item(j, i, 2) - self.meanR[N]), 2)
        
        self.stdB[N] = math.sqrt(self.stdB[N] / (self.h[N]*self.w[N]))
        self.stdG[N] = math.sqrt(self.stdG[N] / (self.h[N]*self.w[N]))
        self.stdR[N] = math.sqrt(self.stdR[N] / (self.h[N]*self.w[N]))

    def fusion(self):
        img = np.zeros((self.h[0],self.w[0],3), np.uint8)
        for j in range(0, self.h[0]):
            for i in range(0, self.w[0]):
                new_Val_b = (self.source_image.item(j, i, 0) - self.meanB[0]) * (self.stdB[1] / self.stdB[0]) + self.meanB[1]
                new_Val_g = (self.source_image.item(j, i, 0) - self.meanG[0]) * (self.stdG[1] / self.stdG[0]) + self.meanG[1]
                new_Val_r = (self.source_image.item(j, i, 0) - self.meanR[0]) * (self.stdR[1] / self.stdR[0]) + self.meanR[1]
                
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

                img.itemset((j, i, 0), new_Val_b)
                img.itemset((j, i, 1), new_Val_g)
                img.itemset((j, i, 2), new_Val_r)
        cv2.imshow("result image", img)
        return img




#####################################################
#                       EXE
#####################################################
        
CT = colorTransfer(FILE_SOURCE, FILE_TARGET)
CT.calMean('source')
CT.calMean('target')
CT.calStd('source')
CT.calStd('target')
CT.fusion()

cv2.imshow("source image", CT.source_image)
cv2.imshow("target image", CT.target_image)
cv2.waitKey(0)
