#!/bin/env python3
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
#import seaborn as sns

startnm=340
endnm=830
interval=10
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
    curve = gausscurve(gauss, 1, hue, 200/sat-199)
#    curve = gausscurve(gauss, 1, hue, lerp(sat, endnm-startnm, 1))
    curve /= curve.sum()
    curve *= bright
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

x_obs = 1.056*gausscurve(gauss_xyz, 599.8, 37.9, 31.0)+0.362*gausscurve(gauss_xyz, 442.0, 16.0, 26.7)-0.065*gausscurve(gauss_xyz, 501.1, 20.4, 26.2)
y_obs = 0.812*gausscurve(gauss_xyz, 568.8, 46.9, 40.5)+0.286*gausscurve(gauss_xyz, 530.9, 16.3, 31.1)
z_obs = 1.217*gausscurve(gauss_xyz, 437.0, 11.8, 36.0)+0.681*gausscurve(gauss_xyz, 459.0, 26.0, 13.8)

r_obs = gausscurve(gauss, 1, 650, 50)
g_obs = gausscurve(gauss, 1, 550, 50)
b_obs = gausscurve(gauss, 1, 450, 50)

num_points = 2000
points_src = np.zeros((num_points, 3))
points_tgt = np.zeros((num_points, 3))
# TODO make these one file
outfile_src = open('src', 'w')
outfile_tgt = open('tgt', 'w')

#for i in tqdm(range(num_points)):
#    c = color(lerp(random.random(), startnm, endnm), random.random(), random.random())
#    points_src[i] = apply_observer(c, x_obs, y_obs, z_obs)
#    points_tgt[i] = apply_observer(c, r_obs, g_obs, b_obs)
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')

# satch test
#for h, s, b in tqdm(np.ndindex(nmcount, 100, 100)):
#n=1000
#colors = np.zeros((n,3))
#sats = np.zeros((n))
#lumas = np.zeros((n))
#
#for i in range(1, n):
#    # sat var
#    colors[i] = color_xyz(650, 1, i/n)
#    lumas[i] = color(650, 1, i/n).sum()
#for i in range(n):
#    s = sat(colors[i])
##    if s == 0: continue
##    sats[i] = math.log(s,2)
#    sats[i] = s

#### STATIC PLOT
#fig, ax = plt.subplots()
##ax.plot(nms, r_obs, scaley=True)
##ax.plot(nms, g_obs, scaley=True)
##ax.plot(nms, b_obs, scaley=True)
##ax.hist(a, bins=100)
#ax.plot(range(1000), sats, label='satch')
#ax.plot(range(1000), lumas, label='luma')
#ax.grid()
#ax.legend()
#plt.show()

### ANIMATED PLOT
fig, ax = plt.subplots()
(lines,) = ax.plot(nms, color(700, 1, 0.01), scaley=False, animated=True)
ax.grid()
plt.show(block=False)

plt.pause(0.1)
bg = fig.canvas.copy_from_bbox(fig.bbox)
ax.draw_artist(lines)
fig.canvas.blit(fig.bbox)

for i in range(1, 100000):
#for i in nmrange:
    fig.canvas.restore_region(bg)

#    lines.set_ydata(color(700, 1, (100-i)/100))
#    lines.set_ydata(color(i, 1, .5))

    #colors[i] = color_xyz(650, 1, i/n)
    c = color(lerp(random.random(), startnm, endnm), random.random(), random.random())
    lines.set_ydata(c)
#    print(i/n, sat(color_xyz(650,1,i/n)))
    #print(colors[i])
    #l = luma(colors[i])
    #if l == 0: continue
    #a = math.log(l,2)

    ax.draw_artist(lines)
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()

plt.pause(1);


