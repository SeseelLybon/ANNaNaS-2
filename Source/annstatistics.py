from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np
import pyglet
from typing import List

import maintools
from maintools import rng
log = maintools.colLogger("tictactoe")


class Statistics:
    def __init__(self):
        self.figure:mpl.figure = mpl.figure.Figure((13,9))
        self.axis:List[any] = list()

        self.axis.append( self.figure.add_subplot(3,2,1, label="Best Score this pop") )
        self.axis.append( self.figure.add_subplot(3,2,3, label="Best Score this gen") )
        self.axis.append( self.figure.add_subplot(3,1,2, label="HistoHeatmap score history") )
        self.axis.append( self.figure.add_subplot(3,1,2, label="HistoHeatmap score species") )

        #self.images:List[pyglet.image.ImageData] = list()

        canvas = FigureCanvasAgg(self.figure)
        data, (w, h) = canvas.print_to_buffer()
        self.image:pyglet.image.ImageData = None


    def update(self, curgen:int, genscoresmax:List[float], genscorescur:List[float], scorehistohist:List[List[float]], scorehistospecies:List[List[float]]):
        self.figure.clear();
        lenaxis = len(self.axis);
        self.axis.clear();

        # Normalize scorehistospecies per row
        for i in range(len(scorehistospecies)):
            rowmax = max(scorehistospecies[i]+[1]);
            scorehistospecies[i] = [ i/rowmax for i in scorehistospecies[i]];
            continue;

        self.axis.append( self.figure.add_subplot(2,3,1, label="Best Score this pop") )
        self.axis.append( self.figure.add_subplot(2,3,4, label="Best Score this gen") )
        self.axis.append( self.figure.add_subplot(1,3,2, label="HistoHeatmap score history") )
        self.axis.append( self.figure.add_subplot(1,3,3, label="HistoHeatmap score species") )

        self.axis[0].plot(range(curgen-len(genscoresmax),curgen), genscoresmax)
        self.axis[0].set_title("Best Score/total pop");
        #self.axis[0].set_xlabel("Generation");
        self.axis[0].set_ylabel("Score");
        self.axis[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        self.axis[1].plot(range(curgen-len(genscorescur),curgen), genscorescur)#, width=1)
        self.axis[1].set_title("Best Score/gen");
        self.axis[1].set_xlabel("Generation");
        self.axis[1].set_ylabel("Score");
        self.axis[1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        self.axis[2].pcolormesh(scorehistohist[::-1], cmap='hot', norm=mpl.colors.Normalize(vmin=0, vmax=500))
        self.axis[2].set_title("Histogram pop/gen");
        self.axis[2].set_xlabel("Score Bin");
        self.axis[2].set_ylabel("Generation");
        #self.axis[2].yaxis.set_major_locator(mpl.ticker.FixedLocator(list(range(curgen, curgen-101, -20))))
        #self.axis[2].yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        self.axis[2].yaxis.set_ticks(list(range(0, 101, 20)))
        self.axis[2].set_yticklabels(list(range(curgen, curgen-101, -20)))

        self.axis[3].pcolormesh(scorehistospecies, cmap='hot')#, norm=mpl.colors.Normalize(vmin=0, vmax=1000))
        self.axis[3].set_title("Histogram score/species");
        self.axis[3].set_xlabel("Score Bin");
        self.axis[3].set_ylabel("Species");
        self.axis[3].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        canvas = FigureCanvasAgg(self.figure)
        data, (w, h) = canvas.print_to_buffer()
        self.image = pyglet.image.ImageData(w, h, "RGBA", data, -4 * w)



if __name__ == "__main__":
    print("Start of statistics")
    from tictactoe import checkWinner
    for i in range(10):
        board = list(rng.integers(0,3, [9])); # create a random board
        while checkWinner(board): # check if it's a valid board (nobody won yet)
            board = list(rng.integers(0,3, [9])); # create a random board

        board[rng.integers(0,9)] = 0; # set a spot to 0 so it can always move
        print(board)
    print("End of statistics")