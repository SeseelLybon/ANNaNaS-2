from __future__ import annotations

import random

import numpy as np
from meeple import Meeple
from population import Population
from typing import List
from typing import Tuple

# A collection of populations that work together.
# The fitness of each part is tracted seperatly, but they are mashed together.
class Ecosystem:
    def __init__(self, pop_size:int, brainputs:List[Tuple[int,int]]):
        self.meepsinPop = pop_size;
        self.symbiot_sizes:List[Tuple[int,int]] = brainputs
        self.populations:List[Population] = [];
        self.generation:int = 0;
        for i in range(len(brainputs)):
            self.populations.append(Population(pop_size, brainputs[i][0], brainputs[i][1]))


    def naturalSelection(self):
        for i in range(len(self.populations)):
            self.populations[i].naturalSelection()

        self.generation += 1;
        return;

    def think(self, meepi:int, vision:List[float]):
        visioni = 0
        visionj = 0
        decisionList:List[float] = list()
        for j in range(0, len(self.populations)-1): # run through all sub-ANN's
            visioni=visionj;
            visionj+=self.symbiot_sizes[j][0];
            self.populations[j].pop[meepi].think(vision[visioni:visionj]);
            decisionList += self.populations[j].pop[meepi].decisionVector;
        self.populations[-1].pop[meepi].think(decisionList+vision[visionj:]);


# Created before use?
class Symbiot:
    def __init__(self):
        self.meeples:List[Meeple] = list();
        self.decisionVector:List[float] = list();


    def think(self):
        self.decisionVector = list();
        for i in range(len(self.meeples)):
            pass

        self.decisionVector = self.meeples[-1].decisionVector;
        pass

    # Specific function for meeples[0]
    # Scores based on how well it spots empty cells
    def fitness_1(self, decision:List[float], board:List[int]):
        runningSum:float = 0
        for i in range(len(board)):
            expected:int = 0;
            if board[i] >= 1:
                expected = 1;
            runningSum += 10/((decision[i] - expected)**2+1)
        return sum

if __name__ == "__main__":
    print("Start of Ecosystem")
    meepsinPop = 10
    ecosystem = Ecosystem(meepsinPop, [(9,9),
                                       (9+9,9)])


    for i in range(10):
        for symbiot_i in range(meepsinPop):
            ecosystem.think(symbiot_i, vision=[1,1,1,1,1,1,1,1,1,
                                               1,1,1,1,1,1,1,1,1])
            print(ecosystem.populations[-1].pop[symbiot_i].decisionVector)

        for popi in range(len(ecosystem.populations)):
            for meepi in range(len(ecosystem.populations[popi].pop)):
                ecosystem.populations[popi].pop[meepi].score = random.uniform(0,10)

        ecosystem.naturalSelection()

    print("End of Ecosystem")
