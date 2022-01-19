#!/bin/env python3
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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
    return gausscurve(gauss, bright, hue, 10/sat-9);

def lerp(x, a, b):
    return a+x*(b-a)

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

#fig, ax = plt.subplots()
#(lines,) = ax.plot(nms, color(700, 1, 0.01), scaley=False, animated=True)
#ax.grid()
#plt.show(block=False)
#
#plt.pause(0.1)
#bg = fig.canvas.copy_from_bbox(fig.bbox)
#ax.draw_artist(lines)
#fig.canvas.blit(fig.bbox)
#
##for i in range(1, 100):
#for i in nmrange:
#    fig.canvas.restore_region(bg)
##    lines.set_ydata(color(700, 1, (100-i)/100))
#    lines.set_ydata(color(i, 1, .5))
#    ax.draw_artist(lines)
#    fig.canvas.blit(fig.bbox)
#    fig.canvas.flush_events()
#
#plt.pause(1);


fig, ax = plt.subplots()

ax.plot(nms, 1.056*gausscurve(gauss_xyz, 599.8, 37.9, 31.0)+0.362*gausscurve(gauss_xyz, 442.0, 16.0, 26.7)-0.065*gausscurve(gauss_xyz, 501.1, 20.4, 26.2), scaley=True)
ax.plot(nms, 0.812*gausscurve(gauss_xyz, 568.8, 46.9, 40.5)+0.286*gausscurve(gauss_xyz, 530.9, 16.3, 31.1), scaley=True)
ax.plot(nms, 1.217*gausscurve(gauss_xyz, 437.0, 11.8, 36.0)+0.681*gausscurve(gauss_xyz, 459.0, 26.0, 13.8), scaley=True)
ax.grid()
plt.show()




