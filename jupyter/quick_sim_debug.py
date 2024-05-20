#!/usr/bin/env python
# coding: utf-8

# In[1]:


# rmsoldie genetic algorithms
import sys
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
from enum import Enum, EnumMeta, auto
from IPython.display import display, Video, HTML, clear_output
from geneticalgorithm import geneticalgorithm as ga


# In[2]:


def clamp(lower, higher, val):
    if val < lower:
        val = lower
    elif val > higher:
        val = higher
    return val


# In[3]:


class Vesicle():
    def __init__(self, x, y, state_arr):
        global general_gap
        # dir: (wavebool, gapsize)
        directions = ["N", "E", "S", "W"]
        self.empty_table = [(direct, False, general_gap) for direct in directions]
        self.input_table = self.empty_table.copy()
        self.coordinates = (x, y)
        self.state_val = 0
        self.grid = state_arr
            
    def get_input_table(self):
        return self.input_table
    
    def set_grid(self, state_arr):
        self.grid = state_arr
    
    def get_state(self):
        return self.state_val
    
    def set_state(self, new_state):
        self.state_val = new_state
        
    def set_gap(self, gap_direct, gap_size):
        for i in range(len(self.input_table)):
            (direct, wb, _) = self.input_table[i]
            if direct == gap_direct:
                self.input_table[i] = (direct, wb, gap_size)
                
    def null_gaps(self):
        self.set_gap("N", 0)
        self.set_gap("E", 0)
        self.set_gap("S", 0)
        self.set_gap("W", 0)
        
    def update(self, state_arr): #problem
        self.calc_inputs(state_arr)
        self.calc_state()
    
    def calc_inputs(self, state_arr):
        global width, height
        (x, y) = self.coordinates
        gaps = []
        for (_, _, gap) in self.input_table:
            gaps.append(gap)
            
        self.input_table = []         
        if (x-1) in range(height) and (y) in range(width):
            north = state_arr[x-1][y].get_state()
        else:
            north = 0
        
        if (x) in range(height) and (y+1) in range(width):
            east = state_arr[x][y+1].get_state()
        else:
            east = 0
        if (x+1) in range(height) and (y) in range(width):
            south = state_arr[x+1][y].get_state()
        else:
            south = 0
        if (x) in range(height) and (y-1) in range(width):
            west = state_arr[x][y-1].get_state()
        else:
            west = 0
            
        if north in [1, 5, 7, 9, 12, 16]:
            tr = True
        else:
            tr = False
        self.input_table.append(("N", tr, gaps[0]))
            
        if east in [2, 5, 6, 8, 11, 16]:
            tr = True
        else:
            tr = False
        self.input_table.append(("E", tr, gaps[1]))
            
        if south in [3, 8, 9, 10, 14, 16]:
            tr = True
        else:
            tr = False
        self.input_table.append(("S", tr, gaps[2]))
        
        if west in [4, 6, 7, 10, 13, 16]:
            tr = True
        else:
            tr = False
        self.input_table.append(("W", tr, gaps[3]))
        
    def calc_state(self):
        global gap_threshold, lookup_table
        inp = self.input_table
        stimed_directs = []
        
        for (direct, wavebool, gapsize) in inp:
            if wavebool == True and gapsize > gap_threshold:
                stimed_directs.append((direct))
        
        if len(stimed_directs) == 1:
            stimed_directs = (stimed_directs[0],)
        else:
            stimed_directs = tuple(stimed_directs)
        self.state_val = lookup_table[stimed_directs]
        


# In[4]:


def build_lookup_tables():
    lookup_table=dict()
    temp = []
    # 0
    temp.append(())
    temp.append(("N",))
    temp.append(("E",))
    temp.append(("S",))
    temp.append(("W",))
    # 5
    temp.append(("N", "E"))
    temp.append(("N", "S"))
    temp.append(("N", "W"))
    temp.append(("E", "S"))
    temp.append(("E", "W"))
    temp.append(("S", "W"))
    # 11
    temp.append(("N", "E", "S"))
    temp.append(("N", "E", "W"))
    temp.append(("N", "S", "W"))
    temp.append(("E", "S", "W"))
    # 15
    temp.append(("N", "E", "S", "W"))
    
    for i in range(len(temp)):
        elem = temp[i]
        lookup_table[elem] = i

    return lookup_table

def stim(x, y):
    global state_arr, width, height
    if x in range(width) and y in range(height):
        state_arr[y][x].set_state(16)
        
def set_gap(x, y, east_wall: bool, gapsize):
    global state_arr, width, height
    if east_wall:
        if x in range(width) and x+1 in range(width) and y in range(height):
            state_arr[y][x].set_gap("E", gapsize)
            state_arr[y][x+1].set_gap("W", gapsize)
        # else - a border
    # south wall
    else:
        if x in range(width) and y in range(height) and y+1 in range(height):
            state_arr[y][x].set_gap("S", gapsize)
            state_arr[y+1][x].set_gap("N", gapsize)
        # else - a border
        
def clear_array(grid):
    for i in range(height):
        for j in range(width):
            grid[i][j].set_state(0)
            grid[i][j].null_gaps()

def clear_states(grid):
    for i in range(height):
        for j in range(width):
            grid[i][j].set_state(0)


# In[5]:


# def get_opposite_dir(dir):
#     opp_dir = (dir+2)%4
#     return opp_dir

# def get_neigh_input(dir, cell_tuple: tuple[Output, list]):
#     dir = get_opposite_dir(dir)
#     _, out_arr = cell_tuple
#     if dir in out_arr:
#         return True
#     else:
#         return False
    
# def calc_new_values(cell_tuple: tuple[Output, list]):
#     # get all 4 neigh values
#     enum, out_arr = cell_tuple
#     if enum in AllOutputs._member_names_:
#         # handle empty cell
#         # handle all outputs
#         pass
    
#     input_dir = enum.value%10
#     # ooh how to handle this...
#     pass

# def update_vesicles(state_arr):
#     num_rows, num_columns = np.shape(state_arr)
#     new_array = []
#     for x in range(num_rows):
#         for y in range(num_columns):
#             # decide new value
                   
            
#             new_array[x, y]
#     return new_array


# In[6]:


import copy

def upd_grid(grid):
    old_grid = copy.deepcopy(grid)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            grid[i][j].update(old_grid)
    return grid

def get_vals_to_print(state_arr):
    num_rows, num_columns = np.shape(state_arr)
    out = ""
    for x in range(num_rows):
        for y in range(num_columns):
            out+=" "+str(state_arr[x][y].get_state()).zfill(2)
        out+="\n"
    return out


# In[7]:


def init_arrays(lowerBound, upperBound, h_in_blocks, w_in_blocks):
  curU = np.zeros((h_in_blocks, w_in_blocks))
  curV = np.zeros((h_in_blocks, w_in_blocks))
  for x in range(h_in_blocks):
    for y in range(w_in_blocks):
      curU[x, y] = round(random.uniform(lowerBound, upperBound), 5)
      curV[x, y] = round(random.uniform(lowerBound, upperBound), 5)
  return curU, curV
  
def init_arrays(lowerBound, upperBound, h_in_blocks, w_in_blocks):
  curU = np.zeros((h_in_blocks, w_in_blocks))
  curV = np.zeros((h_in_blocks, w_in_blocks))
  for x in range(h_in_blocks):
    for y in range(w_in_blocks):
      curU[x, y] = round(random.uniform(lowerBound, upperBound), 5)
      curV[x, y] = round(random.uniform(lowerBound, upperBound), 5)
  return curU, curV
  
def find_neighbours_sum(curU, x, y):
    global w_in_pixels, h_in_pixels
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
    global h_in_pixels, w_in_pixels
    global phi, epsilon, f, q, dt, du
    
    cur_u_grid = image_arr[:, :, 0]
    cur_v_grid = image_arr[:, :, 1]
    illum = image_arr[:, :, 2]
    new_u_grid = np.zeros((h_in_pixels, w_in_pixels))
    new_v_grid = np.zeros((h_in_pixels, w_in_pixels))
    sig_fig = 6

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
            new_u_grid[x, y] = u + ( ((1.0/epsilon) * (u - np.square(u) - ((f*v + phi)*(u - q)/(u + q)))) + laplacian ) * dt
            
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

def square_stimulus(cur_u_grid, y, x):
    global vesicle_size
    half_vesicle=vesicle_size//2 -1
    #  stim 4 at x, y locations!!
    x=(vesicle_size-1)*x + half_vesicle
    y=(vesicle_size-1)*y + half_vesicle
    cur_u_grid[x][y] = 1
    cur_u_grid[x+1][y+1] = 1
    cur_u_grid[x+1][y] = 1
    cur_u_grid[x][y+1] = 1
    return cur_u_grid

def center_stim_detect(grid, y, x):
    global vesicle_size
    cur_u_grid=grid[:,:,0]
    half_vesicle=vesicle_size//2 -1
    #  stim 4 at x, y locations!!
    x=(vesicle_size-1)*x + half_vesicle
    y=(vesicle_size-1)*y + half_vesicle
    if cur_u_grid[x][y] >= 0.4 or cur_u_grid[x+1][y+1] >= 0.4 or cur_u_grid[x+1][y] >= 0.4 or cur_u_grid[x][y+1] >= 0.4:
        return True
    return False
  
def makegrid(slit_len):
    global w_in_pixels, h_in_pixels, width, height
    half_vesicle=(vesicle_size-1)//2
    grid = np.zeros((h_in_pixels, w_in_pixels))
    for i in range(w_in_pixels):
        grid[0, i] = 1
        grid[h_in_pixels-1, i] = 1
    
    for i in range(h_in_pixels):
        grid[i, 0] = 1
        grid[i, w_in_pixels-1] = 1
    
    for h in range(1, height+1):
        for w in range(1, width+1):
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
  
def setgap(illum, w, h, rightside: bool, slit_len):
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
    return illum
            
def assemble_check_grid(best_fit):
    global h_in_pixels, w_in_pixels, slit_len, selective_gap
    cur_u_grid = np.zeros((h_in_pixels, w_in_pixels))
    cur_v_grid = np.zeros((h_in_pixels, w_in_pixels))
    illum = makegrid(slit_len)
    if best_fit[0]:
        illum = setgap(illum, 0, 0, True, selective_gap)
    if best_fit[1]:
        illum = setgap(illum, 1, 0, True, selective_gap)
    if best_fit[2]:
        illum = setgap(illum, 0, 0, False, selective_gap)
    if best_fit[3]:
        illum = setgap(illum, 1, 0, False, selective_gap)
    if best_fit[4]:
        illum = setgap(illum, 2, 0, False, selective_gap)
    if best_fit[5]:
        illum = setgap(illum, 0, 1, True, selective_gap)
    if best_fit[6]:
        illum = setgap(illum, 1, 1, True, selective_gap)
    return cur_u_grid, cur_v_grid, illum


# In[8]:


ga_iterations = 300
hi_fi_check_iterations = 50
check_with_ga=True

total_frames = 10
general_gap = 0
custom_gap=3
gap_threshold = 0
width, height = 3, 2
lookup_table = build_lookup_tables()
state_arr = np.empty((height, width), dtype=object)
for i in range(height):
    for j in range(width):
        state_arr[i][j] = Vesicle(i, j, state_arr)
        
phi = 0.072
# phi = 0.0745
epsilon = 0.0243; f = 1.4; q = 0.002; dt = 0.001; du = 0.45
#  vary width of slits, vesicle size. take constant phi=0.079?
vesicle_size = 16
slit_len = 0
selective_gap = 7
hifi_iterations = 5000
h_in_pixels = height*vesicle_size - (height-1) -1
w_in_pixels = width*vesicle_size - (width-1) -1
check_output_cell_every = 100

# 7 gaps for a 3x2 grid; all binary
# AND gate specific evaluation
def fitness_func(bound_array):
  global state_arr
  fitval=0
  clear_array(state_arr)
  if bound_array[0]:
    set_gap(0, 0, True, custom_gap)
  if bound_array[1]:
    set_gap(1, 0, True, custom_gap)
  if bound_array[2]:
    set_gap(0, 0, False, custom_gap)
  if bound_array[3]:
    set_gap(1, 0, False, custom_gap)
  if bound_array[4]:
    set_gap(2, 0, False, custom_gap)
  if bound_array[5]:
    set_gap(0, 1, True, custom_gap)
  if bound_array[6]:
    set_gap(1, 1, True, custom_gap)

  if bound_array[0]==1 or bound_array[2]==1:
    fitval+=1
  if bound_array[1]==1 or bound_array[4]==1:
    fitval+=1
  if bound_array[3]==1 or bound_array[5]==1 or bound_array[6]==1:
    fitval+=1
  
  # 0, 1 -> 0
  stim(2, 0)
  for frame_count in range(total_frames-1):
    # the output cell got stimulus
    if state_arr[1][1].get_state() > 0:
      # penalty for not adhering to the AND table
      fitval-=5
    state_arr = upd_grid(state_arr)
    
  clear_states(state_arr)
  # 1, 0 -> 0
  stim(0, 0)
  for frame_count in range(total_frames-1):
    # the output cell got stimulus
    if state_arr[1][1].get_state() > 0:
      # penalty for not adhering to the AND table
      fitval-=5
    state_arr = upd_grid(state_arr)
  
  clear_states(state_arr)
  # 1, 1 -> 1
  stim(0, 0)
  stim(2, 0)
  for frame_count in range(total_frames-1):
    # the output cell got stimulus
    if state_arr[1][1].get_state() > 0:
      # +1 so that at max frames points are also awarded
      # it's better than not falling in this if clause
      fitval+=total_frames-frame_count+1
    state_arr = upd_grid(state_arr)

  return -(fitval)

alg_param = {'max_num_iteration': hi_fi_check_iterations,
              'population_size':3,
              'mutation_probability':0.05,
              'elit_ratio': 0,
              'crossover_probability': 0.5,
              'parents_portion': 0.3,
              'crossover_type':'one_point',
              'max_iteration_without_improv':None}

for i in range(0, ga_iterations, hi_fi_check_iterations):
  model = ga(function=fitness_func, dimension=7, variable_type='bool', algorithm_parameters=alg_param)
  model.run()
  best_fit=list(model.output_dict['variable'])

  if check_with_ga:
    # hi-fi check
    cur_u_grid, cur_v_grid, illum = assemble_check_grid(best_fit) 
    # 0, 1 -> 0
    cur_u_grid = square_stimulus(cur_u_grid, 2, 0)
    image_arr = np.stack([cur_u_grid, cur_v_grid, illum], axis = -1)
    for i in range(hifi_iterations):
      if i%check_output_cell_every == 0:
        # wrong output, send back to the GA
        if center_stim_detect(image_arr, 1, 1):
            continue
      image_arr = upd_grid(image_arr)
    
    cur_u_grid, cur_v_grid, illum = assemble_check_grid(best_fit) 
    # 1, 0 -> 0
    cur_u_grid = square_stimulus(cur_u_grid, 0, 0)
    image_arr = np.stack([cur_u_grid, cur_v_grid, illum], axis = -1)
    for i in range(hifi_iterations):
      if i%check_output_cell_every == 0:
        # wrong output, send back to the GA
        if center_stim_detect(image_arr, 1, 1):
            continue
      image_arr = upd_grid(image_arr)
    
    cur_u_grid, cur_v_grid, illum = assemble_check_grid(best_fit) 
    # 1, 1 -> 1
    cur_u_grid = square_stimulus(cur_u_grid, 0, 0)
    cur_u_grid = square_stimulus(cur_u_grid, 2, 0)
    image_arr = np.stack([cur_u_grid, cur_v_grid, illum], axis = -1)
    for i in range(hifi_iterations):
      if i%check_output_cell_every == 0:
        # correct output, successful on the other two, if this part of code is reached
        if center_stim_detect(image_arr, 1, 1):
            hifi_validated = True
            break
      image_arr = upd_grid(image_arr)
      
    if hifi_validated:
      break
    
print(best_fit)

