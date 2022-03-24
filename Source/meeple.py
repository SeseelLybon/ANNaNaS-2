from __future__ import annotations

import trueskill as ts
from typing import List

from neuralnetwork import NeuralNetwork
import elo
import maintools
log = maintools.colLogger("meeple")

class Meeple:
    def __init__(self, braininputs:int, brainoutputs:int, isHollow=False):
        self.score:int=1
        self.fitness:float = 1
        self.unadjustedFitness:float = 0
        self.vision:List[float] = list()
        self.decision:List[float] = list()
        self.lifespan:int = 0
        self.bestScore:int = 0
        self.isAlive:bool=True
        self.brainInputs:int = braininputs
        self.brainOutputs:int = brainoutputs

        if isHollow:
            self.brain = None;
        else:
            self.brain:NeuralNetwork = NeuralNetwork(braininputs, brainoutputs, isHollow)

        self.trueskill = ts.Rating(15)
        self.elo = elo.Rating()
        self.winx:int = 0
        self.wino:int = 0
        self.losex:int = 0
        self.loseo:int = 0
        self.drawx:int = 0
        self.drawo:int = 0
        self.foulx:int = 0
        self.foulo:int = 0


    def show(self)->None:
        return

    def move(self)->None:
        return

    def update(self)->None:
        return

    def look(self, inputValues)->None:
        return

    def think(self, vision, postClean=True)->None:
        #maxi:float = 0
        #maxIndex:int = 0

        self.decision = self.brain.feedForward(vision, postClean=postClean)

        #for deci in range(len(self.decision)):
        #    if self.decision[deci] > max:
        #        max = self.decision[deci]
        #        maxIndex = deci

    def clone(self)->Meeple:
        clone:Meeple = Meeple(self.brainInputs, self.brainOutputs)
        clone.brain = self.brain.clone()
        clone.brain.generateNetwork()
        return clone

    def calculateFitness(self)->None:
        self.fitness = self.score*3;

    def crossover(self, parent2:Meeple)->Meeple:
        child:Meeple = Meeple(self.brainInputs, self.brainOutputs, True)
        child.brain = self.brain.crossover(parent2.brain)
        child.brain.generateNetwork()
        return child