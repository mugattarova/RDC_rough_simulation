# rmsoldie genetic algorithms

import matplotlib.pyplot as plt
import moviepy.editor as mvp
import numpy as np
import os
import random
import time
import tqdm
from matplotlib.colors import to_rgb
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from pathlib import Path
from PIL import Image, ImageDraw, ImageTk

def clamp(lower, higher, val):
    if val < lower:
        val = lower
    elif val > higher:
        val = higher
    return val

def init_arrays(lowerBound, upperBound, h_in_pixels, w_in_pixels):
    curU = np.zeros((h_in_pixels, w_in_pixels))
    curV = np.zeros((h_in_pixels, w_in_pixels))
    for x in range(h_in_pixels):
        for y in range(w_in_pixels):
            curU[x, y] = round(random.uniform(lowerBound, upperBound), 5)
            curV[x, y] = round(random.uniform(lowerBound, upperBound), 5)
    return curU, curV

def get_input_illum(filename):
    vesicle_img = Image.open(filename)
    input_full_arr = np.array(vesicle_img)
    illum = input_full_arr[:, :, 2]
    # where blue, value 255
    return illum

def find_neighbours_sum(curU, x, y):
    h_in_pixels = len(curU[0])
    w_in_pixels = len(curU)
    neighbours_sum = 0
    neighbours = 4
    if x-1 >= 0:
        northN = curU[x-1, y]
        neighbours_sum += northN
    else:
        neighbours -= 1
        
    if y+1 < w_in_pixels:
        eastN = curU[x, y+1]
        neighbours_sum += eastN
    else:
        neighbours -= 1
        
    if x+1 < h_in_pixels:
        southN = curU[x+1, y]
        neighbours_sum += southN
    else:
        neighbours -= 1

    if y-1 >= 0:
        westN = curU[x, y-1]
        neighbours_sum += westN
    else:
        neighbours -= 1   
        
    return neighbours_sum, neighbours

def upd_grid(image_arr):
    cur_u_grid = image_arr[:, :, 0]
    cur_v_grid = image_arr[:, :, 1]
    illum = image_arr[:, :, 2]
    new_u_grid = np.zeros((h_in_pixels, w_in_pixels))
    new_v_grid = np.zeros((h_in_pixels, w_in_pixels))
    sig_fig = 6

    global phi, epsilon, f, q, dt, du

    for x in range(h_in_pixels):
        for y in range(w_in_pixels):
            u = cur_u_grid[x, y]
            v = cur_v_grid[x, y]
            if illum[x, y] != 0:
               new_u_grid[x, y] = 0
               new_v_grid[x, y] = 0
               continue

            neighbours_sum, neighbours = find_neighbours_sum(cur_u_grid, x, y)

            laplacian = du * ( (neighbours_sum - neighbours * cur_u_grid[x, y]) / 0.0625)
            
            # local concentrations of activator
            new_u_grid[x, y] = u + ( ((1.0/epsilon) * (u - np.square(u) - ( (f*v + phi)*(u - q)/(u + q) ))) + laplacian ) * dt
            
            # local concentrations of inhibitor
            new_v_grid[x, y] = v + ( u - v ) * dt

            # # clamp
            # new_u_grid[x, y] = clamp(0.0, 1.0, new_u_grid[x, y])
            # new_v_grid[x, y] = clamp(0.0, 1.0, new_v_grid[x, y])

    np.copyto(cur_u_grid, new_u_grid)
    np.copyto(cur_v_grid, new_v_grid)
    illum = image_arr[:, :, 2]
    new_image_arr = np.stack([cur_u_grid, cur_v_grid, illum], axis = -1)
    return new_image_arr

class VideoWriter:
  def __init__(self, filename, fps=30.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()

def square_stimulus(cur_u_grid, y, x):
    half_vesicle=vesicle_size//2 -1
    #  stim 4 at x, y locations!!
    x=(vesicle_size-1)*x + half_vesicle
    y=(vesicle_size-1)*y + half_vesicle
    cur_u_grid[x][y] = 1
    cur_u_grid[x+1][y+1] = 1
    cur_u_grid[x+1][y] = 1
    cur_u_grid[x][y+1] = 1
    return cur_u_grid

def makegrid(slit_len):
    half_vesicle=(vesicle_size-1)//2
    grid = np.zeros((h_in_pixels, w_in_pixels))
    for i in range(w_in_pixels):
        grid[0, i] = 1
        grid[h_in_pixels-1, i] = 1
    
    for i in range(h_in_pixels):
        grid[i, 0] = 1
        grid[i, w_in_pixels-1] = 1
    
    for h in range(1, h_in_vesicles+1):
        for w in range(1, w_in_vesicles+1):
            top = (h-1)*vesicle_size - (h-1-1) -1
            left = (w-1)*vesicle_size - (w-1-1) -1
            bottom = h*vesicle_size - (h-1) -1
            right = w*vesicle_size - (w-1) -1
            
            if right != w_in_pixels:
                for i in range(top+1, bottom):
                    middle = top+1 + half_vesicle
                    if i in range(middle - slit_len//2, middle + (slit_len - slit_len//2)):
                        pass
                    else:     
                        grid[i, right] = 1
            
            if bottom != h_in_pixels:
                for i in range(left+1, right):
                    middle = left+1 + half_vesicle
                    if i in range(middle - slit_len//2, middle + (slit_len - slit_len//2)):
                        pass
                    else:    
                        grid[bottom, i] = 1
            
            if right!=w_in_pixels and bottom!=h_in_pixels:
                grid[bottom, right] = 1
                if w==1:
                    grid[bottom, left] = 1
                if h==1:
                    grid[top, right] = 1
    return grid

def setgap(w, h, rightside: bool, slit_len):
    global illum
    w+=1
    h+=1
    top = (h-1)*vesicle_size - (h-1-1) -1
    left = (w-1)*vesicle_size - (w-1-1) -1
    bottom = h*vesicle_size - (h-1) -1
    right = w*vesicle_size - (w-1) -1
    
    half_vesicle=(vesicle_size-1)//2
    
    # on the right
    if rightside:
        for i in range(top+1, bottom):
            middle = top+1 + half_vesicle
            if i in range(middle - slit_len//2, middle + (slit_len - slit_len//2)):
                val = 0
            else:     
                val = 1
            illum[i, right] = val
                
    # on the bottom
    else:
        for i in range(left+1, right):
            middle = left+1 + half_vesicle
            if i in range(middle - slit_len//2, middle + (slit_len - slit_len//2)):
                val = 0
            else:    
                val = 1
            illum[bottom, i] = val
    
    return 

# ------------ constants ------------
# Oregonator eq
# phi = 0.079 
# near excitability
# [0.05, 0.08] is good (adamatzky2017fredkin, adamatzky2018street)
#  0.074 between 0.075
# phi = 0.0673
phi = 0.055
epsilon = 0.0243
f = 1.4
q = 0.002
dt = 0.001
du = 0.45

#  vary width of slits, vesicle size. take constant phi=0.079

h_in_vesicles = 4
w_in_vesicles = 3
vesicle_size = 20
slit_len = 0

iterations = 5000
record_every_n =100
record_start_frame = 0

h_in_pixels = h_in_vesicles*vesicle_size - (h_in_vesicles-1) -1
w_in_pixels = w_in_vesicles*vesicle_size - (w_in_vesicles-1) -1

# ------------ main ------------ #
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
input_file=("./input_pics/vesicle_input_"+str(w_in_pixels)+"x"+str(h_in_pixels)+".png")

cur_u_grid = np.zeros((h_in_pixels, w_in_pixels))
cur_v_grid = np.zeros((h_in_pixels, w_in_pixels))
illum = makegrid(slit_len)
opengap = 2
setgap(0, 1, True, opengap)
setgap(1, 0, False, opengap)
setgap(1, 1, True, opengap)
setgap(1, 1, False, 3)

# stimulus 4 adjacent pixels
cur_u_grid = square_stimulus(cur_u_grid, 1, 0)
cur_u_grid = square_stimulus(cur_u_grid, 0, 1)
cur_u_grid = square_stimulus(cur_u_grid, 2, 1)

image_arr = np.stack([cur_u_grid, cur_v_grid, illum], axis = -1)
image_arr = upd_grid(image_arr)
print("shape of image array:", image_arr.shape)
print()

out_fn = "./videos/output.mp4"
Path(out_fn).touch()
with VideoWriter(out_fn) as vid:
    for i in tqdm.trange(iterations):
        if i > record_start_frame:
            if i%record_every_n == 0:
                vid.add(image_arr)
        image_arr = upd_grid(image_arr)
    
