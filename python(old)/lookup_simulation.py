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

def clamp(lower, higher, val):
    if val < lower:
        val = lower
    elif val > higher:
        val = higher
    return val

class Output(Enum):
    pass
class AllOutputs(Output):
    EMPTY = 0
    ALL_OUTPUTS = 1
class NoOutput(Output):
   UP_INPUT = 10
   RIGHT_INPUT = 11
   DOWN_INPUT = 12
   LEFT_INPUT = 13
class SideOutputs(Output):
    UP_INPUT = 20
    RIGHT_INPUT = 21
    DOWN_INPUT = 22
    LEFT_INPUT = 23
class ThreeOutputs(Output):
    UP_INPUT = 30
    RIGHT_INPUT = 31
    DOWN_INPUT = 32
    LEFT_INPUT = 33

def get_opposite_dir(dir):
    opp_dir = (dir+2)%4
    return opp_dir

def get_neigh_input(dir, cell_tuple: tuple[Output, list]):
    dir = get_opposite_dir(dir)
    _, out_arr = cell_tuple
    if dir in out_arr:
        return True
    else:
        return False

def calc_new_values(cell_tuple: tuple[Output, list]):
    # get all 4 neigh values
    enum, out_arr = cell_tuple
    if enum in AllOutputs._member_names_:
        # handle empty cell
        # handle all outputs
        pass
    
    input_dir = enum.value%10
    # ooh how to handle this...
    pass

def update_vesicles(state_arr):
    num_rows, num_columns = np.shape(state_arr)
    new_array = []
    for x in range(num_rows):
        for y in range(num_columns):
            # decide new value
                   
            
            new_array[x, y]
    return new_array

def get_vals_to_print(state_arr: list):
    num_rows, num_columns = np.shape(state_arr)
    out = ""
    for x in range(num_rows):
        for y in range(num_columns):
            out+=" "+state_arr[x, y]
        out+="\n"
    return out

# ------------ constants ------------
# UP = 0
# RIGHT = 1
# DOWN = 2
# LEFT = 3
h_in_vesicles = 3
w_in_vesicles = 3
frames = 5

# ------------ stimulus ------------
stim_row, stim_column = 1, 1

# ------------ main ------------ #
state_arr = np.full((h_in_vesicles, w_in_vesicles), (AllOutputs.EMPTY, []))
state_arr[stim_row, stim_column] = (AllOutputs.ALL_OUTPUTS, [0, 1, 2, 3])

print("shape of image array:", state_arr.shape)

for frame in range(frames):
    state_arr = update_vesicles(state_arr)
    time.sleep(1)
        