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
        self.inputConnections:List[Connection] = list()     # for backprop
        self.layer:int = 0

    def __repr__(self):
        return str(self.ID)

    def fire(self, maxLayer:int) -> None:
        if self.layer == maxLayer:
            self.outputValue = self.Sigmoid(self.inputSum)
        if self.layer != 0:
            self.outputValue = self.ReLU1(self.inputSum)

        for coni in range(len(self.outputConnections)):
            if self.outputConnections[coni].enabled:
                self.outputConnections[coni].toNode.inputSum += self.outputConnections[coni].weight * self.outputValue



    def isConnectedTo(self, node: Node) -> bool:
        """Tests connection between self and node

        Args:
            node (Node): Node to check connection to
        Returns:
            bool: True if connected
        """
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
        if 0 > x:
            x=0;
        return x;
        #return max(0, x) # Highly illigal, using floats in min/max
    @staticmethod
    def ReLU2(x: float) -> float:
        if 0 > x:
            x=0;
        elif x > 1:
            x=1;
        return x;
        # return max(0, min(x, 1)) # Highly illigal, using floats in min/max
    @staticmethod
    def ReLU3(x: float) -> float:
        if -1 > x:
            x=-1;
        elif x > 1:
            x=1;
        return x;
        #return max(-1, min(x, 1)) # Highly illigal, using floats in min/max
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

        if fromNode.layer < toNode.layer:
            self.isRecurrent = False
        else:
            self.isRecurrent = True

        self.weight:float = weight
        self.innovationNumber:int = innovationNo

        self.enabled:bool = True

    def __repr__(self):
        return str(self.innovationNumber)

    def mutateWeight(self):
        if rng.uniform() < .05: # 5% chance to drastically change the weight
            #self.weight = rng.uniform(-1,1)
            self.weight = rng.normal()
        else:# rand1 < .25: # n% chance to slightly change the weight should make it more stable
            self.weight += rng.normal()/10
            #self.weight = np.min([1, np.max([self.weight, -1])])
        min(5, max(-5, self.weight))

    def clone(self, fromNode:Node, toNode:Node) -> Connection:
        temp:Connection = Connection(fromNode, toNode, self.weight, self.innovationNumber)
        temp.enabled = self.enabled
        return temp





import logging
import unittest
class TestNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from logging import DEBUG
        log.logger.setLevel(DEBUG);

    def tearDown(self)->None:
        pass;

    #@unittest.expectedFailure
    #def functioniexpecttofail(self):
    #   pass;

    def testActivation(self):
        test = [2, 1, 0.5, 0, -0.5, -1, -2]

        answer = [2, 1, 0.5, 0, 0, 0, 0]
        with self.subTest("ReLU1"):
            for t, a in zip(test, answer):
                self.assertTrue(Node.ReLU1(t)==a, "%f:%f"%(t,a));

        answer = [1, 1, 0.5, 0, 0, 0, 0]
        with self.subTest("ReLU2"):
            for t, a in zip(test, answer):
                self.assertTrue(Node.ReLU2(t)==a, "%s:%s"%(t,a));

        answer = [1, 1, 0.5, 0, -.5, -1, -1]
        with self.subTest("ReLU3"):
            for t, a in zip(test, answer):
                self.assertTrue(Node.ReLU3(t)==a, "%f:%f"%(t,a));

    def test_Node(self):
        Node1 = Node(1);
        Node2 = Node(2);
        dummyOutconnectiong = Connection(Node1, Node2, 0.5, 1)


        pass;

    def test_Connection(self):
        dummyNode = Node(1);
        connection = Connection(dummyNode,dummyNode,0,1);
        with self.subTest("MutateWeight"):
            connection.mutateWeight();
            self.assertFalse(connection.weight==0);
            self.assertTrue(connection.weight>-1);
            self.assertTrue(connection.weight<1);
        pass;
