#!/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math

est=[]
act=[]

for j in range(60):
    chroma_dim = j
    
    i=0
    
    for x,y,z in np.ndindex((chroma_dim,chroma_dim,chroma_dim)):
#        x /= chroma_dim
#        y /= chroma_dim
#        z /= chroma_dim
        if x+y+z==chroma_dim:
#            print(f'{i:>8} {x:>8.1f} {y:>8.1f} {z:>8.1f}')
            i+=1
    
    #for x,y in np.ndindex((chroma_dim,chroma_dim)):
    #    x /= chroma_dim
    #    y /= chroma_dim
    #    z = 1-x-y
    #    if x+y+z==1:
    #        print(x,y,z)
    #        i+=1

    est.append(j**3)
    act.append(i)
    #print(f'{j},{j**3},{i}')
    print(f'{j} {i} {math.sqrt(i)}')
    
#fig,ax = plt.subplots()
#ax.plot(est, act, 'ro')
#ax.plot([x for x in range(1000000)], [x**(79/128) for x in range(1000000)])
#ax.grid()
#ax.legend()
#plt.show()

