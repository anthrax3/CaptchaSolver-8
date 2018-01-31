# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:34:25 2018

@author: b2002032064079
"""

import requests
import random
from PIL import Image
from io import BytesIO
import os

#url_captcha = 'http://web.trf3.jus.br/consultas/Captcha/GerarCaptcha?'
#
#session = requests.Session()
#
#
#for i in range(10000,20000):
#    page = session.get(url_captcha+str(random.uniform(0,1)))
#    img = Image.open(BytesIO(page.content))
#    #img.show()
#    img.save(os.path.join(os.getcwd(), 'captcha_solver', 'samples', str(i+1) + '.jpeg'))


path = 'C:\\Users\\b2002032064079\\Desktop\\CaptchaSolver\\captcha_solver\\separated_letters'

str_letters = 'abcdefghijklmnopqrstuvwxyz'

print(type(str_letters[0]))

for c in str_letters:
    print(c)
    if not os.path.exists(os.path.join(path,c)):
        os.makedirs(os.path.join(path,c))


    