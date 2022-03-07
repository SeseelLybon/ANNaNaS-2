from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pyglet
from typing import List

from numpy.random import default_rng
rng = default_rng()


class Statistics:
    def __init__(self):
        self.figure:mpl.figure = mpl.figure.Figure((12,8))
        self.axis:List[mpl.axes._subplots.AxesSubplot] = list()

        self.axis.append( self.figure.add_subplot(3,2,1, label="Best Score this pop") )
        self.axis.append( self.figure.add_subplot(3,2,3, label="Best Score this gen") )
        self.axis.append( self.figure.add_subplot(3,1,2, label="HistoHeatmap score history") )
        self.axis.append( self.figure.add_subplot(3,1,2, label="HistoHeatmap score species") )

        #self.images:List[pyglet.image.ImageData] = list()

        canvas = FigureCanvasAgg(self.figure)
        data, (w, h) = canvas.print_to_buffer()
        self.image:pyglet.image.ImageData = None


    def update(self, curgen, genscoresmax:List[float], genscorescur:List[float], scorehistohist:List[List[float]], scorehistospecies:List[List[float]]):
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

        self.axis[0].step(range(curgen-1000,curgen), genscoresmax, linewidth=2.5)
        self.axis[1].step(range(curgen-100,curgen), genscorescur, linewidth=2.5)
        self.axis[2].pcolormesh(scorehistohist[::-1], cmap='hot', norm=mpl.colors.Normalize(vmin=0, vmax=1000))
        self.axis[3].pcolormesh(scorehistospecies, cmap='hot')#, norm=mpl.colors.Normalize(vmin=0, vmax=1000))
        #, norm=colors.Normalize(vmin=Z.min(), vmax=Z.max())
        #self.axis.yticks(range(curgen-100,curgen))

        canvas = FigureCanvasAgg(self.figure)
        data, (w, h) = canvas.print_to_buffer()
        self.image = pyglet.image.ImageData(w, h, "RGBA", data, -4 * w)



def update(dt):
    global genscores, stats, curgen
    curgen+=2
    print("update ", len(genscores))
    genscores2 = genscores[:]
    genscores[:] = genscores[-9:]
    genscores.append(genscores[-1]+rng.integers(1,4));
    stats.update(genscores, genscores2, curgen);
    return;


if __name__ == "__main__":
    window = pyglet.window.Window(1200,800)
    print("Start of statistics")
    stats = Statistics(window)
    curgen = 0
    genscores = [0 for i in range(10)];


    pyglet.clock.schedule_interval_soft(update, 1)
    pyglet.app.run()

    print("End of statistics")