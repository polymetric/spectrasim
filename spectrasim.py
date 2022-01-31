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

pps = 16
#num_points = pps**3
num_points = 96*36*4
points_src = np.zeros((num_points, 3))
points_tgt = np.zeros((num_points, 3))

### random gaussians test
#print('random gaussians')
#outfile_src = open('random_primaries_src', 'w')
#outfile_tgt = open('random_primaries_tgt', 'w')
#random.seed(0)
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
#print('uniform gaussians')
#outfile_src = open('uniform_gaussians_src', 'w')
#outfile_tgt = open('uniform_gaussians_tgt', 'w')
#random.seed(0)
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
#print('uniform hues')
#outfile_src = open('uniform_hue_src', 'w')
#outfile_tgt = open('uniform_hue_tgt', 'w')
#random.seed(0)
#i = 0
#for h in tqdm(range(num_points)):
#    c = color(map(h, 0, num_points, startnm, endnm), 1, 1)
#
#    points_src[i] = apply_observer(c, x_obs, y_obs, z_obs)
#    points_tgt[i] = apply_observer(c, r_obs, g_obs, b_obs)
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')
#    i += 1

### random fixed primaries test
#print('random fixed primaries')
#outfile_src = open('random_fixed_primaries_src', 'w')
#outfile_tgt = open('random_fixed_primaries_tgt', 'w')
#random.seed(0)
#for i in tqdm(range(num_points)):
#    #r=max(0, logc_decode(random.random()))
#    #g=max(0, logc_decode(random.random()))
#    #b=max(0, logc_decode(random.random()))
#    r=max(0, random.random())
#    g=max(0, random.random())
#    b=max(0, random.random())
#    c=color(630, r, 1) + color(532, g, 1) + color(467, b, 1)
#
#    points_src[i] = logc_encode(apply_observer(c, x_obs, y_obs, z_obs))
#    points_tgt[i] = logc_encode(apply_observer(c, r_obs, g_obs, b_obs))
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')

test1=[]

### random fixed monochromes test
#print('random fixed monochromes')
#outfile_src = open('random_fixed_monochromes_src', 'w')
#outfile_tgt = open('random_fixed_monochromes_tgt', 'w')
#random.seed(0)
#for i in tqdm(range(num_points)):
#    r=max(0, logc_decode(random.random()))
#    o=max(0, logc_decode(random.random()))
#    y=max(0, logc_decode(random.random()))
#    g=max(0, logc_decode(random.random()))
#    b=max(0, logc_decode(random.random()))
#    test1.append(logc_encode(r))
#    c=color(622+random.randint(-100, 100), r, 1) + color(605+random.randint(-100, 100), o, 1) + color(591+random.randint(-100, 100), y, 1) + color(568+random.randint(-100, 100), g, 1) + color(462+random.randint(-100, 100), b, 1)
#    c*=max(0, logc_decode(random.random()))
#
#    points_src[i] = logc_encode(apply_observer(c, x_obs, y_obs, z_obs))
#    points_tgt[i] = logc_encode(apply_observer(c, r_obs, g_obs, b_obs))
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')

### random shifting primaries test
#print('random shifting primaries')
#outfile_src = open('random_shifting_primaries_src', 'w')
#outfile_tgt = open('random_shifting_primaries_tgt', 'w')
#random.seed(0)
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

test2=[]

### uniform fixed monochromes
#print('uniform fixed monochromes')
#outfile_src = open('uniform_fixed_monochromes_src', 'w')
#outfile_tgt = open('uniform_fixed_monochromes_tgt', 'w')
#pps=8
#num_points=pps**5
#points_src = np.zeros((num_points, 3))
#points_tgt = np.zeros((num_points, 3))
#random.seed(0)
#i = 0
#for r, o, y, g, b in tqdm(np.ndindex((pps,pps,pps,pps,pps)), total=pps**5):
#    r = max(0, logc_decode(map(r, 0, pps-1, 0, 1)))
#    o = max(0, logc_decode(map(o, 0, pps-1, 0, 1)))
#    y = max(0, logc_decode(map(y, 0, pps-1, 0, 1)))
#    g = max(0, logc_decode(map(g, 0, pps-1, 0, 1)))
#    b = max(0, logc_decode(map(b, 0, pps-1, 0, 1)))
#    test2.append(logc_encode(r))
#    c=color(622, r, 1) + color(605, o, 1) + color(591, y, 1) + color(568, g, 1) + color(462, b, 1)
#
#    points_src[i] = logc_encode(apply_observer(c, x_obs, y_obs, z_obs))
#    points_tgt[i] = logc_encode(apply_observer(c, r_obs, g_obs, b_obs))
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')
#    i += 1

### uniform fixed primaries
#print('uniform fixed primaries')
#outfile_src = open('uniform_fixed_primaries_src', 'w')
#outfile_tgt = open('uniform_fixed_primaries_tgt', 'w')
#random.seed(0)
#i = 0
#for r, g, b in tqdm(np.ndindex((pps,pps,pps)), total=pps**3):
#    #r = max(0, logc_decode(map(r, 0, pps, 0, 1)))
#    #g = max(0, logc_decode(map(g, 0, pps, 0, 1)))
#    #b = max(0, logc_decode(map(b, 0, pps, 0, 1)))
#    r = max(0, map(r, 0, pps, 0, 1)))
#    g = max(0, map(g, 0, pps, 0, 1)))
#    b = max(0, map(b, 0, pps, 0, 1)))
#    c=color(630, r, 1) + color(532, g, 1) + color(467, b, 1)
#
#    points_src[i] = logc_encode(apply_observer(c, x_obs, y_obs, z_obs))
#    points_tgt[i] = logc_encode(apply_observer(c, r_obs, g_obs, b_obs))
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')
#    i += 1

### chroma-log fixed primaries
print('uniform chroma-log fixed primaries (clfp)')
stops = 20
luma_interval = 1/3
chroma_dim = 50
chromas = []
for r,g,b in np.ndindex((chroma_dim,chroma_dim,chroma_dim)):
    if r+g+b == chroma_dim:
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
    if r>=1 and b==0 and g==0: print('beef')
    c=color(630, r, 1) + color(532, g, 1) + color(467, b, 1)
    c *= 0.18*(2**(-(stop*luma_interval)+(stops/2)))

    points_src[i] = logc_encode(np.clip(apply_observer(c, x_obs, y_obs, z_obs), 0, None))
    points_tgt[i] = logc_encode(np.clip(apply_observer(c, r_obs, g_obs, b_obs), 0, None))
    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')
    i += 1

### uniform recovery
#print('uniform recovery')
#outfile_src = open('uniform_recovery_src', 'w')
#outfile_tgt = open('uniform_recovery_tgt', 'w')
#random.seed(0)
#i = 0
#for r, g, b in tqdm(np.ndindex((pps,pps,pps)), total=pps**3):
#    r = max(0, logc_decode(r/15))
#    g = max(0, logc_decode(g/15))
#    b = max(0, logc_decode(b/15))
#    c = colour.recovery.XYZ_to_sd_Jakob2019([r,g,b])
#    c = c.align(colour.SpectralShape(startnm, endnm, interval))
#    c = c.values
#
#    points_src[i] = logc_encode(np.clip(apply_observer(c, x_obs, y_obs, z_obs), 0, None))
#    points_tgt[i] = logc_encode(np.clip(apply_observer(c, r_obs, g_obs, b_obs), 0, None))
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')
#    i += 1

### uniform shifting dual monochromes
#print('uniform shifting dual monochromes')
#outfile_src = open('uniform_shifting_dual_monochromes_src', 'w')
#outfile_tgt = open('uniform_shifting_dual_monochromes_tgt', 'w')
#alumas=2
#blumas=2
#nwaves=2
##waves=[350,463,520,523,569,589,591,605,623,651,655]
#waves=[map(i, 0, nwaves-1, 400, 700) for i in range(nwaves)]
#num_points = alumas*blumas*len(waves)**2
#points_src = np.zeros((num_points, 3))
#points_tgt = np.zeros((num_points, 3))
#i=0
#for la, lb, ha, hb in tqdm(np.ndindex((alumas, blumas, len(waves), len(waves))), total=num_points):
##    if la == lb: continue
#    if ha == hb: continue
#    c = color(waves[ha], 2**-(la+(la/2)), 1) + color(waves[hb], map(lb, 0, blumas-1, 0, logc_decode(1)), 1)
#
#    points_src[i] = logc_encode(apply_observer(c, x_obs, y_obs, z_obs))
#    points_tgt[i] = logc_encode(apply_observer(c, r_obs, g_obs, b_obs))
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')
#    i += 1

#extra_colors = [color(0,0,0), color(0,logc_decode(1),0)]
#for c in extra_colors:
#    points_src[i] = logc_encode(apply_observer(c, x_obs, y_obs, z_obs))
#    points_tgt[i] = logc_encode(apply_observer(c, r_obs, g_obs, b_obs))
#    outfile_src.write(f'{points_src[i][0]} {points_src[i][1]} {points_src[i][2]}\n')
#    outfile_tgt.write(f'{points_tgt[i][0]} {points_tgt[i][1]} {points_tgt[i][2]}\n')

### STATIC PLOT
#fig, ax = plt.subplots()
#
#s = color(0, 0, 1)
#for w in waves:
#    s += color(w, 1, 1)
#
#ax.plot(nms, s, 'y', scaley=False)
#
#ax.plot(nms, r_obs, 'r', scaley=True)
#ax.plot(nms, g_obs, 'g', scaley=True)
#ax.plot(nms, b_obs, 'b', scaley=True)


#paint = color(650, 1.0, .5)
#light = color(450, 0.5, .5)
#
#ax.plot(nms, paint, 'r', scaley=False)
#ax.plot(nms, light, 'b', scaley=False)
#ax.plot(nms, paint*light, 'g', scaley=False)

#ax.plot(nms, x_obs, 'r', scaley=True)
#ax.plot(nms, y_obs, 'g', scaley=True)
#ax.plot(nms, z_obs, 'b', scaley=True)

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
#i=startnm
#
#print('uniform fixed monochromes')
#pps=6
#num_points=pps**5
#points_src = np.zeros((num_points, 3))
#points_tgt = np.zeros((num_points, 3))
#random.seed(0)
#i = 0
#it = np.ndindex((pps,pps,pps,pps,pps))
#
#def update(frame):
#    global i
##    s = color(i, 1, 1)
##    c = apply_observer(s, x_obs, y_obs, z_obs)
##    line1.set_data(nms, x_obs)
##    line2.set_data(nms, y_obs)
##    line3.set_data(nms, z_obs)
##    line4.set_data(nms, s)
##    print(f'{i:>8} {c[0]:>12.8f} {c[1]:>12.8f} {c[2]:>12.8f}')
##    i+=1
##    if i > endnm: i = startnm
#    r,o,y,g,b = it.__next__()
#    r = max(0, logc_decode(map(r, 0, pps, 0, 1)))
#    o = max(0, logc_decode(map(o, 0, pps, 0, 1)))
#    y = max(0, logc_decode(map(y, 0, pps, 0, 1)))
#    g = max(0, logc_decode(map(g, 0, pps, 0, 1)))
#    b = max(0, logc_decode(map(b, 0, pps, 0, 1)))
#
##    r = max(0, logc_decode(random.random()))
##    o = max(0, logc_decode(random.random()))
##    y = max(0, logc_decode(random.random()))
##    g = max(0, logc_decode(random.random()))
##    b = max(0, logc_decode(random.random()))
#    c=color(622, r, 1) + color(605, o, 1) + color(591, y, 1) + color(568, g, 1) + color(462, b, 1)
#
#    line1.set_data(nms, c)
#    line2.set_data(nms, r_obs)
#    line3.set_data(nms, g_obs)
#    line4.set_data(nms, b_obs)
#    print(logc_encode(apply_observer(c, x_obs, y_obs, z_obs)), logc_encode(apply_observer(c, r_obs, g_obs, b_obs)))
#    i += 1
#    return line1,line2,line3,line4,line5,line6,line7
#
#ani = FuncAnimation(fig, update, blit=True, interval=100)
#plt.show()

