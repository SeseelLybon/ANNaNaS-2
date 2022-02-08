from __future__ import annotations

import numpy as np
from numpy.random import default_rng
rng = default_rng()

from typing import List

from neuralnetwork import NeuralNetwork
from neuralnetwork import ConnectionHistory

from meeple import Meeple

class Species:
    def __init__(self, speciesID, meep:Meeple  ):
        self.speciesID = speciesID
        self.bestMeeple:Meeple = meep
        self.meeples:list = [meep]

        self.staleness = 0 # stagnation
        self.fitnessSum = 0
        self.averageFitness = 0
        self.bestFitness = 0

        self.exceesCoeff:float = 1
        self.weightDiffCoeff:float = 1
        self.compatibilityThreshold:float = 4




    def checkSameSpecies(self, meep:Meeple ) -> bool:
        excessAndDisjoint:float = self.getExcessDisjoint(meep.brain, self.bestMeeple.brain)
        averageWeightDiff:float = self.averageWeightDiff(meep.brain, self.bestMeeple.brain)

        largeConnectionsNormaliser = len(meep.brain.connections) - 20
        if largeConnectionsNormaliser < 1:
            largeConnectionsNormaliser = 1

        compatibility:float = (self.exceesCoeff*excessAndDisjoint/largeConnectionsNormaliser)+(self.weightDiffCoeff*averageWeightDiff)
        return self.compatibilityThreshold > compatibility

    def addToSpecies(self, meep:Meeple):
        self.meeples.append(meep)

    def getExcessDisjoint(self, brain1:NeuralNetwork, brain2:NeuralNetwork) -> float:
        matching:float = 0

        for conni in range(len(brain1.connections)):
            for connj in range(len(brain2.connections)):
                if brain1.connections[conni].innovationNumber == brain2.connections[connj].innovationNumber:
                    matching+=1
                    break


        return len(brain1.connections) + len(brain2.connections) - (2 * matching)

    # Returns the average weight diffirence between meeples.
    def averageWeightDiff(self, brain1:NeuralNetwork, brain2:NeuralNetwork) -> float:
        if (len(brain1.connections) == 0) or (len(brain2.connections) == 0 ):
            return 0

        matching:float = 0
        totalDiff:float = 0

        for conni in range(len(brain1.connections)):
            for connj in range(len(brain2.connections)):
                if brain1.connections[conni].innovationNumber == brain2.connections[connj].innovationNumber:
                    matching+=1
                    totalDiff += np.absolute(brain1.connections[conni].weight - brain2.connections[connj].weight)
                    break
        if matching == 0:
            return 100

        return totalDiff/matching

    def sortSpecies(self):
        self.meeples.sort(key=lambda meep: meep.fitness, reverse=True)

        if self.bestFitness < self.meeples[0].fitness:
            self.bestFitness = self.meeples[0].fitness
            self.bestMeeple = self.meeples[0]
            self.staleness=0
        else:
            self.staleness+=1

    def generateChild(self,innovationHistory:List[ConnectionHistory])->Meeple:
        child:Meeple=None

        if rng.random() < 0.25:
            child = self.selectParent().clone()
        else:
            parent1:Meeple = self.selectParent()
            parent2:Meeple = self.selectParent()
            if parent1.fitness > parent2.fitness:
                child = parent1.crossover(parent2)
            else:
                child = parent2.crossover(parent1)

        child.brain.mutate(innovationHistory)

        return child



    def setFitnessSum(self) -> None:
        self.fitnessSum = 0
        for i in range(len(self.meeples)):
            self.fitnessSum += self.meeples[i].fitness

    def setAverageFitness(self):
        runningSum = 0
        for i in range(len(self.meeples)):
            runningSum += self.meeples[i].fitness

        self.averageFitness = runningSum/len(self.meeples)

    def selectParent(self) -> Meeple:
        fitnessSum = 0
        for i in range(len(self.meeples)):
            fitnessSum += self.meeples[i].fitness
        rand = rng.uniform(0, self.fitnessSum)

        runningSum = 0

        for i in range(len(self.meeples)):
            runningSum +=  self.meeples[i].fitness
            if runningSum > rand:
                return self.meeples[i]

        return self.meeples[0]

    def cull(self) -> None:
        #self.sortSpecie()
        if len(self.meeples) > 2:
            self.meeples[:] = self.meeples[:len(self.meeples)//2]

    def fitnessSharing(self)->None:
        for meep in range(len(self.meeples)):
            self.meeples[meep].fitness/=len(self.meeples)

if __name__ == "__main__":
    import timeit
    import cProfile

    print("Starting species.py as main")

    # p = cProfile.Profile()
    # p.runctx('oldbrain.ReLU(x)', locals={'x': 5}, globals={'oldbrain':oldbrain} )
    # p.runcall(oldbrain.fire_network)
    # p.print_stats()

    print("Finished species.py as main")


