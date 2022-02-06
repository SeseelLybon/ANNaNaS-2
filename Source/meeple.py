from __future__ import annotations

from neuralnetwork import NeuralNetwork

from typing import List

class Meeple:
    def __init__(self, braininputs:int, brainoutputs:int, isHollow=False):
        self.fitness:float;
        self.unadjustedFitness:float
        self.vision:List[float] = list()
        self.decision:List[float] = list()
        self.lifespan:int = 0
        self.bestScore:int = 0
        self.dead:bool
        self.score:int
        self.generation = 0
        self.brainInputs:int = braininputs
        self.brainOutputs:int = brainoutputs

        self.brain = NeuralNetwork(braininputs, brainoutputs, isHollow)

    def show(self)->None:
        return

    def move(self)->None:
        return

    def update(self)->None:
        return

    def look(self)->None:
        return

    def think(self)->None:
        max:float = 0
        maxIndex:int = 0

        self.decision = self.brain.feedForward(vision)

        for deci in range(len(self.decision)):
            if self.decision[deci] > max:
                max = self.decision[deci]
                maxIndex = deci

    def clone(self)->Meeple:
        clone:Meeple = Meeple(self.brainInputs, self.brainOutputs)
        clone.brain = self.brain.clone()
        clone.fitness = self.fitness
        clone.brain.generateNetwork()
        clone.generation = self.generation
        clone.bestScore = self.bestScore
        return clone

    def calculateFitness(self)->None:
        self.fitness = self.score**2

    def crossover(self, parent2:Meeple)->Meeple:
        child:Meeple = Meeple(self.brainInputs, self.brainOutputs, True)
        child.brain = self.brain.crossover(paren2.brain)
        child.brain.generateNetwork()
        return child