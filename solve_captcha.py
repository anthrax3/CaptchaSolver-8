# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:34:25 2018

@author: b2002032064079
"""

import requests
import random
from PIL import Image
from io import BytesIO

url_captcha = 'http://web.trf3.jus.br/consultas/Captcha/GerarCaptcha?'

session = requests.Session()


for i in range(10000):
    page = session.get(url_captcha+str(random.uniform(0,1)))
    img = Image.open(BytesIO(page.content))
    #img.show()
    img.save('samples//'+str(i+1)+'.jpeg')
    