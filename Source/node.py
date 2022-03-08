from __future__ import annotations

import math
import numpy as np


from typing import List
from typing import NoReturn

import maintools
from maintools import rng
log = maintools.colLogger("node")

class Node:
    def __init__(self, ID:int):
        self.ID:int = ID
        self.inputSum:float = 0
        self.outputValue:float = 0
        self.outputConnections:List[Connection] = list()
        self.layer:int = 0

    def __repr__(self):
        return str(self.ID)

    def fire(self, maxLayer:int) -> None:
        if self.layer == maxLayer:
            self.outputValue = self.ReLU1(self.inputSum)
        if self.layer != 0:
            self.outputValue = self.ReLU1(self.inputSum)

        for coni in range(len(self.outputConnections)):
            if self.outputConnections[coni].enabled:
                self.outputConnections[coni].toNode.inputSum += self.outputConnections[coni].weight * self.outputValue



    def isConnectedTo(self, node:Node) -> bool:
        if node.layer == self.layer:
            return False

        if node.layer < self.layer:
            for nodei in range(len(node.outputConnections)):
                if node.outputConnections[nodei].toNode == self:
                    return True
        else:
            for nodei in range(len(self.outputConnections)):
                if self.outputConnections[nodei].toNode == node:
                    return True

        return False


    def clone(self) -> Node:
        clone:Node = Node(self.ID)
        clone.layer = self.layer
        return clone


    @staticmethod
    def Sigmoid(x: float) -> float:
        return 1 / (1 + math.e ** -x)
    @staticmethod
    def ReLU1(x: float) -> float:
        return np.max([0, x])
    @staticmethod
    def ReLU2(x: float) -> float:
        return max(0, min(x, 1))
    @staticmethod
    def ReLU3(x: float) -> float:
        return max(-1, min(x, 1))

    @staticmethod
    def ReLU4(x: float) -> int:
        if x <= 0:
            return 0
        else:
            return 1

    @staticmethod
    def Cosine(x: float) -> float:
        return math.cos(x)
    @staticmethod
    def Sine(x: float) -> float:
        return math.sin(x)



#Here because of circular dependancy
class Connection:
    def __init__(self, fromNode:Node, toNode:Node, weight:float, innovationNo:int):
        self.fromNode:Node = fromNode
        self.toNode:Node = toNode

        self.weight:float = weight
        self.innovationNumber:int = innovationNo

        self.enabled:bool = True

    def __repr__(self):
        return str(self.innovationNumber)

    def mutateWeight(self):
        rand1:float = rng.random()
        if rand1 < .01: # 1% chance to drastically change the weight
            #self.weight = rng.uniform(-1,1)
            self.weight = rng.normal()
        else:# rand1 < .25: # 90% chance to slightly change the weight should make it more stable
            self.weight += rng.normal()/50
            #self.weight = np.min([1, np.max([self.weight, -1])])
        min(1, max(-1, self.weight))

    def clone(self, fromNode:Node, toNode:Node) -> Connection:
        temp:Connection = Connection(fromNode, toNode, self.weight, self.innovationNumber)
        temp.enabled = self.enabled
        return temp



if __name__ == "__main__":

    import timeit
    import cProfile
    print("Starting node.py as main")

    #p = cProfile.Profile()
    #p.runctx('oldbrain.ReLU(x)', locals={'x': 5}, globals={'oldbrain':oldbrain} )
    #p.runcall(oldbrain.fire_network)
    #p.print_stats()

    print("Finished node.py as main")