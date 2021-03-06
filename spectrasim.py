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

# TODO vectorize?
def matchtonemap(x, src, tgt):
    if type(x) == np.ndarray:
        for i in x.flat:
            matchtonemap(i)
        return x

    i0=0
    i1=0
    if x <= src[0]:
        if extrapolate:
            i0=1
            i1=2
        else:
            return src[0]
    else:
        for i in range(len(src)):
            if i == len(src)-1:
                if extrapolate:
                    i0 = i-1
                    i1 = i
                    break
                else:
                    return tgt[i]

            if src[i] <= x and x <= table_src[i+1]:
                i0 = i
                i1 = i+1
                break
    
    x0 = src[i0]
    x1 = src[i1]
    y0 = tgt[i0]
    y1 = tgt[i1]

    return lerp(x, x0, y0, x1, y1)

x_obs = 1.056*gausscurve(gauss_xyz, 599.8, 37.9, 31.0)+0.362*gausscurve(gauss_xyz, 442.0, 16.0, 26.7)-0.065*gausscurve(gauss_xyz, 501.1, 20.4, 26.2)
y_obs = 0.812*gausscurve(gauss_xyz, 568.8, 46.9, 40.5)+0.286*gausscurve(gauss_xyz, 530.9, 16.3, 31.1)
z_obs = 1.217*gausscurve(gauss_xyz, 437.0, 11.8, 36.0)+0.681*gausscurve(gauss_xyz, 459.0, 26.0, 13.8)

### random cursed rgb function
#r_obs = gausscurve(gauss, 1, 700, 50)
#g_obs = gausscurve(gauss, 1, 550, 50)
#b_obs = gausscurve(gauss, 1, 380, 50)

### cie RGB
#r_obs = x_obs *  0.41847    + y_obs * -0.15866    + z_obs * -0.082835
#g_obs = x_obs * -0.09169    + y_obs *  0.25243    + z_obs *  0.015708
#b_obs = x_obs *  0.00092090 + y_obs * -0.0025498  + z_obs *  0.17870  

### 709 
#r_obs = x_obs *  3.2404542  + y_obs * -1.5371385  + z_obs * -0.4985314
#g_obs = x_obs * -0.9692660  + y_obs *  1.8760108  + z_obs *  0.0415560
#b_obs = x_obs *  0.0556434  + y_obs * -0.2040259  + z_obs *  1.0572252
#
#r_obs = np.clip(r_obs, 0, None)
#g_obs = np.clip(g_obs, 0, None)
#b_obs = np.clip(b_obs, 0, None)

# load ssf from file
t=load_table('5219 ssf.txt')
r_obs = {}
g_obs = {}
b_obs = {}
for nm, b in zip(range(380,731,10), t):
    r_obs[nm] = b[0]
    g_obs[nm] = b[1]
    b_obs[nm] = b[2]
r_obs = colour.SpectralDistribution(r_obs)
g_obs = colour.SpectralDistribution(g_obs)
b_obs = colour.SpectralDistribution(b_obs)

ekw = {'method':'Constant', 'left':0, 'right':0}

r_obs.align(colour.SpectralShape(startnm, endnm, interval), extrapolator=colour.Extrapolator, extrapolator_kwargs=ekw)
g_obs.align(colour.SpectralShape(startnm, endnm, interval), extrapolator=colour.Extrapolator, extrapolator_kwargs=ekw)
b_obs.align(colour.SpectralShape(startnm, endnm, interval), extrapolator=colour.Extrapolator, extrapolator_kwargs=ekw)

r_obs = r_obs.values/1566.575376
g_obs = g_obs.values/1566.575376
b_obs = b_obs.values/1566.575376



# load densities from file
t=load_table('5219 dens.txt')
y_dens = {}
m_dens = {}
c_dens = {}
for nm, b in zip(range(380,731,10), t):
    y_dens[nm] = b[0]
    m_dens[nm] = b[1]
    c_dens[nm] = b[2]
y_dens = colour.SpectralDistribution(y_dens)
m_dens = colour.SpectralDistribution(m_dens)
c_dens = colour.SpectralDistribution(c_dens)

ekw = {'method':'Constant', 'left':0, 'right':0}

y_dens.align(colour.SpectralShape(startnm, endnm, interval), extrapolator=colour.Extrapolator, extrapolator_kwargs=ekw)
m_dens.align(colour.SpectralShape(startnm, endnm, interval), extrapolator=colour.Extrapolator, extrapolator_kwargs=ekw)
c_dens.align(colour.SpectralShape(startnm, endnm, interval), extrapolator=colour.Extrapolator, extrapolator_kwargs=ekw)

y_dens = y_dens.values
m_dens = m_dens.values
c_dens = c_dens.values

### chroma-log fixed primaries
print('uniform chroma-log fixed primaries (clfp)')
stops = 20
luma_interval = 1/3
chroma_dim = 50
chromas = []
for r,g,b in np.ndindex((chroma_dim,chroma_dim,chroma_dim)):
    if r+g+b == chroma_dim-1:
        r/=chroma_dim-1
        g/=chroma_dim-1
        b/=chroma_dim-1
        chromas.append([r,g,b])
chromas = np.array(chromas)
num_points = stops*int(1/luma_interval)*chromas.shape[0]
points_src = np.zeros((num_points, 3))
points_tgt = np.zeros((num_points, 3))
outfile_src = open('uniform_clfp_src', 'w')
outfile_tgt = open('uniform_clfp_tgt', 'w')
i = 0
for stop, chroma_i in tqdm(np.ndindex((stops*int(1/luma_interval),chromas.shape[0])), total=num_points):
    chroma = chromas[chroma_i]
    r=chroma[0]
    g=chroma[1]
    b=chroma[2]
    c=color(630, r, 1) + color(532, g, 1) + color(467, b, 1)
    c *= 0.18*(2**(-(stop*luma_interval)+(stops/2)))

    points_src[i] = logc_encode(np.clip(apply_observer(c, x_obs, y_obs, z_obs), 0, None))
    #points_tgt[i] = np.clip(apply_observer(c, r_obs, g_obs, b_obs), 0, None)
    # film emulation
    c = np.clip(apply_observer(c, r_obs, g_obs, b_obs), 0, None)
    c = matchtonemap(c, neg_curve_src, neg_curve_tgt)
    s = np.full(nmcount, 1.0)
    s *= (y_dens-1)*(1-np.clip(c[0], 0, 1))+1
    s *= (m_dens-1)*(1-np.clip(c[1], 0, 1))+1
    s *= (c_dens-1)*(1-np.clip(c[2], 0, 1))+1
    c = logc_encode(np.clip(apply_observer(s, x_obs, y_obs, z_obs), 0, None)*2**-10)
    points_tgt[i] = c
    # end film emulation
    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')
    i += 1

### STATIC PLOT
#fig, ax = plt.subplots()
#
#ax.plot(nms, r_obs, 'r', scaley=True)
#ax.plot(nms, g_obs, 'g', scaley=True)
#ax.plot(nms, b_obs, 'b', scaley=True)

#ax.plot(nms, x_obs, 'r', scaley=True)
#ax.plot(nms, y_obs, 'g', scaley=True)
#ax.plot(nms, z_obs, 'b', scaley=True)


#ax.grid()
#ax.legend()
#plt.show()


### ANIMATED PLOT
#fig, ax = plt.subplots()
#figcolor, axcolor = plt.subplots()
#(line1,) = ax.plot(nms, color(0, 0, 0), 'y', scaley=False, animated=True)
#(line2,) = ax.plot(nms, color(0, 0, 0), 'r', scaley=False, animated=True)
#(line3,) = ax.plot(nms, color(0, 0, 0), 'g', scaley=False, animated=True)
#(line4,) = ax.plot(nms, color(0, 0, 0), 'b', scaley=False, animated=True)
#(line5,) = ax.plot(nms, color(0, 0, 0), 'r', scaley=False, animated=True)
#(line6,) = ax.plot(nms, color(0, 0, 0), 'g', scaley=False, animated=True)
#(line7,) = ax.plot(nms, color(0, 0, 0), 'b', scaley=False, animated=True)
#ax.grid()
#ax.set_ylim(0, 2)
#
#def update(frame):
#    return line1,line2,line3,line4,line5,line6,line7
#
#ani = FuncAnimation(fig, update, blit=True, interval=100)
#plt.show()

