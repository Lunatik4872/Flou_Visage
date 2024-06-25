from PIL import Image
import numpy as np
from pandas import *

def get_neighbours(np_picture, x, y, size):
    neighbours = []
    coo = []

    for xp in range(max(0,y-size),min(len(np_picture),y+size+1)) :
        for yp in range(max(0,x-size),min(len(np_picture[0]),x+size+1)) :
            if (yp,xp) != (x,y) : 
                neighbours.append(np_picture[xp][yp])
                coo.append((xp,yp))

    return neighbours,coo

def blur(np_picture,x,y) :
    neighbours,coo = get_neighbours(np_picture,x,y,50)

    # propagation of the pixel's colour to its neighbours
    np_picture[y][x] = np.mean(neighbours, axis=0).astype(int)
    for nei in coo : np_picture[nei[0]][nei[1]] = np_picture[y][x]

    return np_picture
