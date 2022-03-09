from __future__ import annotations


import colorlog
import logging
import time

class colLogger:
    def __init__(self, name, level=logging.INFO):
        handler = colorlog.StreamHandler();
        handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(name)s:%(message)s'));
        self.logger = colorlog.getLogger(name);
        self.logger.addHandler(handler);
        self.logger.setLevel(level);

log = colLogger("maintools");

from numpy.random import default_rng
rng = default_rng()


class loadingBar:
    def __init__(self, maxgames:int, steps:int):
        self.maxgames:int = maxgames;
        self.curgame:int = 0;
        self.steps:int = steps;
        self.gamestep:int = maxgames//steps;

    def loadingBarIncrement(self)->None:
        self.curgame+=1
        if self.curgame%self.gamestep==0:
            print("=",end="")

    def printLoadingbar(self)->None:
        time.sleep(0.3)
        lbarstr:str = ""
        for dummy in range(self.steps//5):
            lbarstr+="    -";
        for dummy in range((self.steps%5)-1):
            lbarstr+=" ";
        lbarstr += "| %d / %d / %d"%(self.maxgames,self.gamestep,self.steps)
        #log.logger.info(lbarstr)
        print(lbarstr)

loadingbar = loadingBar(1000, 20);