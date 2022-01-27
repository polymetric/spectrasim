#!/bin/env python3
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import colour
from colour.plotting import *

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

x_obs = 1.056*gausscurve(gauss_xyz, 599.8, 37.9, 31.0)+0.362*gausscurve(gauss_xyz, 442.0, 16.0, 26.7)-0.065*gausscurve(gauss_xyz, 501.1, 20.4, 26.2)
y_obs = 0.812*gausscurve(gauss_xyz, 568.8, 46.9, 40.5)+0.286*gausscurve(gauss_xyz, 530.9, 16.3, 31.1)
z_obs = 1.217*gausscurve(gauss_xyz, 437.0, 11.8, 36.0)+0.681*gausscurve(gauss_xyz, 459.0, 26.0, 13.8)

r_obs = gausscurve(gauss, 1, 700, 50)
g_obs = gausscurve(gauss, 1, 550, 50)
b_obs = gausscurve(gauss, 1, 380, 50)

#num_points = 4096
pps = 16
num_points = pps**3
points_src = np.zeros((num_points, 3))
points_tgt = np.zeros((num_points, 3))
# TODO make these one file
outfile_src = open('uniform_hues_src', 'w')
outfile_tgt = open('uniform_hues_tgt', 'w')
random.seed(0)

### random gaussians test
#for i in tqdm(range(num_points)):
#    c = color(0, 0.0, 0)
#    for j in range(random.randint(0, 3)):
#        c += color(lerp(random.random(), startnm, endnm), max(0, logc_decode(random.random())), 1)
#
#    points_src[i] = logc_encode(apply_observer(c, x_obs, y_obs, z_obs))
#    points_tgt[i] = logc_encode(apply_observer(c, r_obs, g_obs, b_obs))
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')

### uniform gaussians test
#i = 0
#for h, s, b in tqdm(np.ndindex((pps,pps,pps)), total=pps**3):
#    c = color(map(h, 0, pps, startnm, endnm), max(0, logc_decode(map(b, 0, pps-1, 0, 1))), s/pps)
#
#    points_src[i] = logc_encode(apply_observer(c, x_obs, y_obs, z_obs))
#    points_tgt[i] = logc_encode(apply_observer(c, r_obs, g_obs, b_obs))
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')
#    i += 1

### uniform hues
#i = 0
#for h in tqdm(range(num_points)):
#    c = color(map(h, 0, num_points, startnm, endnm), 1, 1)
#
#    points_src[i] = apply_observer(c, x_obs, y_obs, z_obs)
#    points_tgt[i] = apply_observer(c, r_obs, g_obs, b_obs)
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')
#    i += 1

### random primaries test
#for i in tqdm(range(num_points)):
#    r=max(0, logc_decode(random.random()))
#    g=max(0, logc_decode(random.random()))
#    b=max(0, logc_decode(random.random()))
#    c=color(700, r, 1) + color(520, g, 1) + color(380, b, 1)
#
#    points_src[i] = logc_encode(apply_observer(c, x_obs, y_obs, z_obs))
#    points_tgt[i] = logc_encode(apply_observer(c, r_obs, g_obs, b_obs))
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')

### random shifting primaries test
#for i in tqdm(range(num_points)):
#    r=max(0, logc_decode(random.random()))
#    g=max(0, logc_decode(random.random()))
#    b=max(0, logc_decode(random.random()))
#    c=color(700+random.randint(-100, 100), r, 1) + color(520+random.randint(-100, 100), g, 1) + color(380+random.randint(-100, 100), b, 1)
#
#    points_src[i] = logc_encode(apply_observer(c, x_obs, y_obs, z_obs))
#    points_tgt[i] = logc_encode(apply_observer(c, r_obs, g_obs, b_obs))
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')

### uniform primaries
#i = 0
#for r, g, b in tqdm(np.ndindex((pps,pps,pps)), total=pps**3):
#    r = max(0, logc_decode(map(r, 0, pps, 0, 1)))
#    g = max(0, logc_decode(map(g, 0, pps, 0, 1)))
#    b = max(0, logc_decode(map(b, 0, pps, 0, 1)))
#    c=color(700, r, 1) + color(550, g, 1) + color(450, b, 1)
#
#    points_src[i] = logc_encode(apply_observer(c, x_obs, y_obs, z_obs))
#    points_tgt[i] = logc_encode(apply_observer(c, r_obs, g_obs, b_obs))
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')
#    i += 1

### uniform recovery primaries
#i = 0
#for r, g, b in tqdm(np.ndindex((pps,pps,pps)), total=pps**3):
#    r = max(0, logc_decode(r/15))
#    g = max(0, logc_decode(g/15))
#    b = max(0, logc_decode(b/15))
#    c = colour.recovery.XYZ_to_sd_Jakob2019([r,g,b])
#    c = c.align(colour.SpectralShape(startnm, endnm, interval))
#    c = c.values
#
#    points_src[i] = logc_encode(apply_observer(c, x_obs, y_obs, z_obs))
#    points_tgt[i] = logc_encode(apply_observer(c, r_obs, g_obs, b_obs))
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')
##    outfile_test.write(f'{logc_encode(r)} {logc_encode(g)} {logc_encode(b)}\n')
#    i += 1

### STATIC PLOT
#fig, ax = plt.subplots()



#ax.plot(nms, r_obs, scaley=True)
#ax.plot(nms, g_obs, scaley=True)
#ax.plot(nms, b_obs, scaley=True)


#paint = color(650, 1.0, .5)
#light = color(450, 0.5, .5)
#
#ax.plot(nms, paint, 'r', scaley=False)
#ax.plot(nms, light, 'b', scaley=False)
#ax.plot(nms, paint*light, 'g', scaley=False)

#ax.plot(nms, x_obs, 'r', scaley=False)
#ax.plot(nms, y_obs, 'g', scaley=False)
#ax.plot(nms, z_obs, 'b', scaley=False)

#ax.hist(a, bins=100)
#ax.plot(range(1000), sats, label='satch')
#ax.plot(range(1000), lumas, label='luma')
#ax.hist(points_src, bins=100)


### recovery test
#s = color(650, 1.0, 1)
##s+= color(550, 1.0, 1)
#c = apply_observer(s, x_obs, y_obs, z_obs)
##c = np.array([1,0,0])
##c /= np.max(c)
#print(c/np.max(c))
#sr = colour.recovery.XYZ_to_sd_Jakob2019(c)
#sr = sr.align(colour.SpectralShape(startnm, endnm, interval))
#sr = sr.values
#c2 = apply_observer(sr, x_obs, y_obs, z_obs)
##c2 /= np.max(c2)
#print(c2/np.max(c2))
#colour_style()
##plot_single_sd(sr)
#
#ax.plot(nms, s, 'r', scaley=False)
#ax.plot(nms, sr, 'b', scaley=False)




#ax.grid()
#ax.legend()
#plt.show()


### ANIMATED PLOT
fig, ax = plt.subplots()
(lines1,) = ax.plot(nms, color(0, 0, 0), 'r', scaley=False, animated=True)
(lines2,) = ax.plot(nms, color(0, 0, 0), 'b', scaley=False, animated=True)
ax.grid()
plt.show(block=False)

plt.pause(0.1)
bg = fig.canvas.copy_from_bbox(fig.bbox)
ax.draw_artist(lines1)
ax.draw_artist(lines2)
fig.canvas.blit(fig.bbox)

for i in range(1000):
#for i in nmrange:
    fig.canvas.restore_region(bg)

#    lines.set_ydata(color(700, 1, (100-i)/100))
#    lines.set_ydata(color(i, 1, .5))

    #colors[i] = color_xyz(650, 1, i/n)
#    c = color(lerp(random.random(), startnm, endnm), random.random(), random.random())
#    lines.set_ydata(c)
#    print(i/n, sat(color_xyz(650,1,i/n)))
    #print(colors[i])
    #l = luma(colors[i])
    #if l == 0: continue
    #a = math.log(l,2)

#    print(2**map(b,0,12,-12,0))
#    s = color(550, 2**map(b, 0, 13-1, -13-1, 0), 1)
#    c = apply_observer(s,x_obs, y_obs, z_obs)
#    lines.set_ydata(s)



#    ### recovery test
#    s = color(map(b,0,100,400,650), 1.0, .8)
#    c = apply_observer(s, x_obs, y_obs, z_obs)
#    c = c/c.sum()
#    print(c)
#    sr = colour.recovery.XYZ_to_sd_Jakob2019(c)
#    sr = sr.extrapolate(colour.SpectralShape(startnm, endnm, interval))
#    sr = sr.interpolate(colour.SpectralShape(startnm, endnm, interval))
#    sr = sr.values
#    colour_style()
#    #plot_single_sd(sr)
#    
#    lines1.set_ydata(s)
#    lines2.set_ydata(sr)


    c = color(map(i, 0, num_points, startnm, endnm), 1, 1)

    a = apply_observer(c, x_obs, y_obs, z_obs)
    b = apply_observer(c, r_obs, g_obs, b_obs)

    lines1.set_ydata(c)
#    lines2.set_ydata(b)
    print(a)




    ax.draw_artist(lines1)
#    ax.draw_artist(lines2)
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()
    plt.pause(0.1);

plt.pause(1);









