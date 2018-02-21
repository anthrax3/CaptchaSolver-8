# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:52:28 2018

@author: B15599226
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

dir_input = "C:/Users/b15599226\Documents/CaptchaSolver/captcha_solver/samples_letters/"
file_names = os.listdir(dir_input)
file_name2 = "C:/Users/b15599226/Documents/CaptchaSolver/captcha_solver/samples_letters/letter_34_4.jpeg"
max_1 = 0
max_2 = 0
y_list = []
x_list = []
for file_name in file_names:
    img = cv2.imread(os.path.join(dir_input, file_name))
    y_list.append(img.shape[0])
    x_list.append(img.shape[1])
    if img.shape[0] > max_1 and img.shape[1] > max_2:
        max_1 =  img.shape[0]   
        max_2 = img.shape[1] 

print(max_1, max_2)
y_list = np.array(y_list)
# the histogram of the data
mu, sigma = np.mean(y_list), np.std(y_list)

n, bins, patches = plt.hist(y_list, 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y_list, 'r--', linewidth=1)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.axis([10, 30, 0, 30])
plt.grid(True)

plt.show()



x_list = np.array(x_list)
# the histogram of the data
mu, sigma = np.mean(x_list), np.std(x_list)

n, bins, patches = plt.hist(x_list, 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, x_list, 'r--', linewidth=1)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.axis([10, 30, 0, 30])
plt.grid(True)

plt.show()