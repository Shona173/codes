import numpy as np
import scipy as sp
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image,ImageDraw
import glob

from scipy.spatial.distance import cdist

from sklearn import manifold
from sklearn.decomposition import PCA

import ot 
import sys 


def draw_square(draw, position, size, color):
    draw.rectangle([position, (position[0] + size, position[1] + size)], fill=color)

frame1 = Image.new("RGB", (800, 600), (255, 255, 255))
draw1 = ImageDraw.Draw(frame1)
draw_square(draw1, (100, 100), 200, (255, 0, 0))

frame2 = Image.new("RGB", (800, 600), (255, 255, 255))
draw2 = ImageDraw.Draw(frame2)
draw_square(draw2, (200, 200), 200, (0, 0, 255))

alpha = 0.5  
blended_frame = Image.blend(frame1,frame2, alpha)

blended_frame.show()
