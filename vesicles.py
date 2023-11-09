import numpy as np
import random
import time
import pygame
from pygame.locals import *

BLACK = (0, 0, 0)
WHITE = (0xFF, 0xFF, 0xFF)
# used in Adamatzky, 2011, Fig.1
blockSize = 25
heightInBlocks = 10
widthInBlocks = 45
WINDOW_HEIGHT = heightInBlocks * blockSize
WINDOW_WIDTH = widthInBlocks * blockSize

currentUGrid = np.zeros((heightInBlocks, widthInBlocks))
currentVGrid = np.zeros((heightInBlocks, widthInBlocks))
newUGrid = np.zeros((heightInBlocks, widthInBlocks))
newVGrid = np.zeros((heightInBlocks, widthInBlocks))

def clamp(lower, higher, val):
    if val < lower:
        val = lower
    elif val > higher:
        val = higher
        
    return val
        
def initArrays(curU, curV, lowerBound, upperBound):
    for x in range(heightInBlocks):
        for y in range(widthInBlocks):
            curU[x, y] = round(random.uniform(lowerBound, upperBound), 5)
            curV[x, y] = round(random.uniform(lowerBound, upperBound), 5)

def updateValuesGrid(currentUGrid, currentVGrid, newUGrid, newVGrid):
    for x in range(heightInBlocks):
        for y in range(widthInBlocks):
            # Oregonator eqs
            # update every dt secs

            epsilon = 0.0243
            phi = 0.079 # near excitability
            f = 1.4
            q = 0.002
            dt = 0.001
            #dx = 0.25
            Du = 0.45 # diffusion coefficient

            u = currentUGrid[x, y]
            v = currentVGrid[x, y]

            neighboursSum = 0
            neighbours = 4
            if x-1 >= 0:
                northN = currentUGrid[x-1, y]
                neighboursSum += northN
            else:
                neighbours -= 1
                
            if y+1 < widthInBlocks:
                eastN = currentUGrid[x, y+1]
                neighboursSum += eastN
            else:
                neighbours -= 1
                
            if x+1 < heightInBlocks:
                southN = currentUGrid[x+1, y]
                neighboursSum += southN
            else:
                neighbours -= 1

            if y-1 >= 0:
                westN = currentUGrid[x, y-1]
                neighboursSum += westN
            else:
                neighbours -= 1   

            # local concentrations of activator
            laplasian = Du * ( (neighboursSum - neighbours * currentUGrid[x, y]) / 0.0625)

            newUGrid[x, y] = currentUGrid[x, y] + ( (1/epsilon) * (u - np.square(u) - ((f*v + phi)*(u - q)/(u + q))) + laplasian ) * dt


            # local concentrations of inhibitor
            newVGrid[x, y] = currentVGrid[x, y] + ( u - v ) * dt

    
    np.copyto(currentUGrid, newUGrid)
    np.copyto(currentVGrid, newVGrid)

def updateGraphicsGrid(currentUGrid, currentVGrid, newUGrid, newVGrid):
    for x in range(heightInBlocks):
        for y in range(widthInBlocks):
            uVal = round( 0xFF * clamp(0.0, 1.0, currentUGrid[x, y]) )
            vVal = round( 0xFF * clamp(0.0, 1.0, currentVGrid[x, y]) )
            pygame.draw.circle(SCREEN, (uVal, vVal, 0), (y*blockSize+(blockSize/2), x*blockSize+(blockSize/2)), blockSize/2)

pygame.init()
SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
CLOCK = pygame.time.Clock()
SCREEN.fill(BLACK)
initArrays(currentUGrid, currentVGrid, 0, 1)

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    pygame.display.update()

    updateValuesGrid(currentUGrid, currentVGrid, newUGrid, newVGrid)
    updateGraphicsGrid(currentUGrid, currentVGrid, newUGrid, newVGrid)

    pygame.display.flip()
    CLOCK.tick(1)

