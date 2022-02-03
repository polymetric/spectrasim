#!/bin/env python3
import tkinter
import numpy
import scipy
import random
import colour
import math
import re

width=1920
height=980

m=tkinter.Tk()
c=tkinter.Canvas(m, width=width, height=height)
c.pack()
c.create_rectangle(100,100,200,200,fill='#000f00')
m.mainloop()

