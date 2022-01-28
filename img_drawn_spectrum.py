#!/bin/env python3
import PIL
import numpy as np
from PIL import Image
image = Image.open('asdfg.png')
spectra = np.zeros(50)

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

for i in range(len(spectra)):
    l = 0
    x = int(map(i, 0, len(spectra)-1, 0, image.width-1))
    for y in range(image.height):
        l+=1-image.getpixel((x,y))
    l/=image.height
    spectra[i] = l

print(spectra)

def lerp(x, a, b):
    return a+x*(b-a)

