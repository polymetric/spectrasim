#!/bin/env python3
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import random
import colour
from colour.plotting import *
import re

startnm=340
endnm=830
interval=1
nmrange = range(startnm, endnm+1, interval);
nmcount = (endnm-startnm)//interval+1
nms = np.array(nmrange)

def gauss(x, amp, offset, width):
    return amp*math.exp((-(x-offset)**2)/(2*width**2))

def gauss_xyz(x, u, o1, o2):
    if x < u:
        return math.exp(-1/2*(x-u)**2/o1**2)
    if x >= u:
        return math.exp(-1/2*(x-u)**2/o2**2)

def gausscurve(g, a, b, c):
    result = np.zeros(nmcount)
    for i, nm in zip(range(len(result)), nmrange):
        result[i] = g(nm, a, b, c)
    return result

def color(hue, bright, sat):
    if (sat == 0): return np.full(nmcount, bright)
    curve = gausscurve(gauss, bright, hue, 200/sat-199)
    return curve;

def lerp(x, a, b):
    return a+x*(b-a)

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def apply_observer(c, r_obs, g_obs, b_obs):
    r = (c * r_obs).sum()
    g = (c * g_obs).sum()
    b = (c * b_obs).sum()
    return np.array([r,g,b])

def color_xyz(hue, bright, sat):
    c = color(hue, bright, sat)
    return apply_observer(c, x_obs, y_obs, z_obs)

def sat(c):
    max_color = np.max(c)
    min_color = np.min(c)
    if max_color == 0: return 0
    return (max_color - min_color) / (max_color + min_color)

def luma(c):
    return np.max(c)

def logc_encode(x):
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809
    mask = np.where(x > cut, 1, 0)
    result = mask * (c * np.log10(a*x+b)+d)
    result += (1-mask) * (e * x + f)
    return result

def logc_decode(x):
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809
    mask = np.where(x > e * cut + f, 1, 0)
    result = mask*(((10**((x-d)/c))-b)/a)
    result += (1-mask) * ((x-f)/e)
    return result

def load_table(file):
    file = open(file, 'r').read()
    list = []
    for i in file.split('\n'):
        row = []
        if i == '': continue
        for j in re.split(' |\t', i):
            row.append(float(j))
        list.append(row)
    return np.array(list)

chroma_dim = 50
chromas = []
for r,g,b in np.ndindex((chroma_dim,chroma_dim,chroma_dim)):
    if r+g+b == chroma_dim-1:
        r/=chroma_dim-1
        g/=chroma_dim-1
        b/=chroma_dim-1
        chromas.append([r,g,b])
chromas = np.array(chromas)


i = 0
outfile = open('chromas', 'w')
for c in tqdm(chromas):
    i += 1
    c /= np.max(c)
    outfile.write(f'{c[0]} {c[1]} {c[2]}\n')




