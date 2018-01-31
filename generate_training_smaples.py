# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:53:07 2018

@author: b2002032064079
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
        

def pre_process_image(filename):
    
    #img = cv2.imread('1.jpeg')
    img = cv2.imread(filename)
#    plt.imshow(img)
#    plt.show()
#    
    img_smt = cv2.GaussianBlur(img,(3,3),0)
 
    imgray = cv2.cvtColor(img_smt, cv2.COLOR_BGR2GRAY)
#    plt.imshow(imgray, cmap='gray')
#    plt.show()
    

    
    #ret, thresh = cv2.threshold(imgray, 200, 255, 0) #127
    ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret, thresh = cv2.threshold(imgray,100,255,cv2.THRESH_BINARY)
    
#    print("{} {}".format(thresh,ret) )
#    (m,n) = thresh.shape
#    plt.imshow(thresh, cmap='gray')
#    plt.show()
    


    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(~thresh, cv2.MORPH_OPEN, kernel)
    
    
#    plt.imshow(opening, cmap='gray')
#    plt.show()

        
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    img, areas = draw_contours(img, imgray, filename, contours, hierarchy)
    
#    plt.imshow(img)
#    plt.show()
#    keyList = list(areas.keys())
#    keyList.sort()
#    print(keyList)
#    for key in keyList:
#        print("'%s': '%s'," % (key, areas[key]), end=" ")   
#    
#    print("")
    
    return areas
    
#    cv2.imshow('Output', img)
#    cv2.waitKey(0)
    
#    plt.imshow(im2,cmap='gray')
#    plt.show()
    
def draw_contours(img, imgray, filename, contours, hierarchy):
    
    try: hierarchy = hierarchy[0]
    except: hierarchy = []
    
    height, width, _= img.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    
    # computes the bounding box for the contour, and draws it on the frame,
    aux = {}
    
    dst_path = os.path.join(os.path.dirname(filename),'..','samples_letters')
    
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    
    base_name = os.path.basename(filename).split('.')[0]
    
    count = 1
    
    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        
        

        if hier[3] == -1:
            cv2.rectangle(img, (x-1,y-1), (x+w+1,y+h+1), (255, 0, 0), 1)
            aux[x] = cv2.contourArea(contour)
            
            if aux[x] >= 50 and aux[x] <= 190:
                aux_name = 'letter_' + base_name + '_' + str(count) + '.jpeg'
                
                count += 1
                
                if not os.path.isfile(os.path.join(dst_path, aux_name)):
                    cv2.imwrite(os.path.join(dst_path, aux_name), imgray[y-1:y+h+1,x-1:x+w+1])
                    
            #areas.append(cv2.contourArea(contour))
        
    
#            
    return img, aux



if __name__ == "__main__":
    
    path = os.path.join(os.getcwd(),'captcha_solver','samples')
    
    lista_areas = []
    

    
    for filename in os.listdir(path):
        lista_areas += list(pre_process_image(os.path.join(path,filename)).values())

    
#    #print(lista_areas)
#    
#    arr_areas = np.array(lista_areas)
#    
##    plt.plot(arr_areas,'r*')
##    plt.show()
##    
#    # the histogram of the data
#    mu, sigma = np.mean(arr_areas), np.std(arr_areas)
#    
#    n, bins, patches = plt.hist(arr_areas, 50, normed=1, facecolor='green', alpha=0.75)
#    
#    # add a 'best fit' line
#    y = mlab.normpdf(bins, mu, sigma)
#    l = plt.plot(bins, y, 'r--', linewidth=1)
#    
#    plt.xlabel('Smarts')
#    plt.ylabel('Probability')
#    plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
#    plt.axis([40, 160, 0, 0.03])
#    plt.grid(True)
#    
#    plt.show()






















